# LJ atom-pair 路径性能优化计划

> 范围约束：保持当前 `ATOM_GROUP` 近邻表语义不变，不引入 GROMACS 风格的 cluster-pair 邻居表。在现有 atom-pair 路径上，尽可能提升 SPONGE 的 LJ / direct-Coulomb GPU 性能。

## 目标

在不改近邻表外部接口的前提下，提升 SPONGE 在 GPU 上的非键相互作用吞吐，重点优化标准 LJ / direct-Coulomb 热路径。

## 边界条件

- 不修改 `ATOM_GROUP` 的数据语义。
- 不修改对外暴露的近邻表接口。
- 不在本分支中引入 cluster-pair、super-cluster 或新型 pairlist 语义。
- 允许修改 LJ 模块内部 kernel、launch 配置、局部缓存策略和写回方式。
- 每一次性能优化改动后，都必须重新测试，不允许凭感觉判断“更快了”。

## 当前基线结论

结合之前的 Nsight Systems / Nsight Compute 结果，当前 LJ 路径的主要问题不是寄存器溢出，而是以下几点：

- `SM Throughput` 偏低，约 `15.66%`
- `Memory Throughput` 偏低，约 `15.59%`
- `Achieved Occupancy` 偏低，约 `42.09%`
- cache 局部性较差：`L2 hit rate ~34.80%`，`L1/TEX hit rate ~0%`
- warp stall 主要集中在：
  - `wait`
  - `short_scoreboard`
  - `branch_resolving`
  - `drain`

这意味着当前瓶颈更偏向：

1. 线程组织不理想
2. 随机访存较多
3. `atomicAdd(frc[atom_j])` 写回压力较大
4. `j` 粒子数据复用不足

## 已验证结论

已经做过一次尝试：将 `sqrt + powf` 改写为 `dr2 / inv_r / inv_r2 / inv_r6` 路径，并对 `water_160k` 做了前后 benchmark 对照。

结果：

- 该改写没有带来可见加速
- 在 `NVT, 20000 steps, warmup=1, repeats=1` 的 quick benchmark 上，`ns/day` 反而约下降 `0.34%`
- 同时这类改写会降低代码可读性

结论：

- 该方向不应继续作为当前阶段的优先优化项
- 文档后续计划中不再把“算术改写”列为主线

## 重点源码位置

- 标准 LJ kernel：
  - `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- LJ 参数与 helper：
  - `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`
- 水分子 fast path：
  - `SPONGE/Lennard_Jones_force/solvent_LJ.cpp`
- 软核 LJ 相关：
  - `SPONGE/Lennard_Jones_force/LJ_soft_core.h`
- 外部 benchmark 工具：
  - `/home/ylj/SPONGE_GITHUB/SPG_GMX_TEST/runner.py`

## 优化主线

### P0：Launch Configuration Sweep

这是当前最应该先做的优化项。

#### 目的

验证当前 LJ kernel 的 block 组织是否显著限制了 occupancy、issue active 和 latency hiding。

#### 原因

- 当前 profiler 结果显示 `occupancy` 和 `issue active` 都不理想
- 现有 kernel 属于明显的 memory-irregular 路径
- 当前 launch 形状较重，可能导致每个 SM 的活跃 block 数不足
- 这类优化不改算法语义，也不明显降低代码可读性

#### 目标文件

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`

#### 执行内容

至少测试以下 launch 配置：

- `(32, 4, 1)`
- `(32, 8, 1)`
- `(32, 16, 1)`
- `(64, 4, 1)`，如果当前映射逻辑允许

保留规则：

- 只有在 end-to-end benchmark 中有稳定收益的配置才能保留
- 不能只看单次 `ncu`，必须看最终 `ns/day`

### P1：降低 `atomicAdd(frc[atom_j])` 写回压力

这是当前最可能带来实质收益的优化方向之一。

#### 原因

当前标准 LJ kernel 中，对 `atom_j` 的力回写采用随机原子写：

- 写回位置离散
- 可能发生较多竞争
- 会增加 LSU 压力
- 会拉低 warp issue 效率

这与当前 `ncu` 中偏低的 memory throughput 和 issue active 是一致的。

#### 目标文件

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- 如有必要，同步评估 `SPONGE/Lennard_Jones_force/solvent_LJ.cpp`

#### 执行内容

优先考虑以下方向：

1. warp 内先局部累加，再做合并写回
2. block 内对相同 `atom_j` 的贡献做局部归并
3. 降低每对 pair 都触发一次全局原子写的频率

#### 约束

- 不改变力学语义
- 不改变近邻表语义
- 不引入新的全局数据结构依赖

### P1：标准 LJ kernel 的 `j` 粒子 tile staging

这是与当前 cache 问题最直接对应的优化项。

#### 原因

当前 general LJ kernel 中，每次 pair 计算都存在较明显的随机读取：

- `atom_j`
- `crd[atom_j]`
- 类型信息
- 电荷信息
- LJ 参数表

这与当前较差的 `L2` / `L1` 命中率高度一致。

#### 目标文件

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`

参考实现：

- `SPONGE/Lennard_Jones_force/solvent_LJ.cpp`

#### 执行内容

1. 将一批 `atom_j` 索引搬入 shared memory
2. 将对应的 `x/y/z/q/type` 搬入 shared memory 或寄存器
3. 在 block 内尽量复用这一 tile
4. 在不改 neighbor-list 语义的前提下改善访存局部性

#### 注意

- 这里只允许改 kernel 调度与临时缓存
- 不允许把 neighbor-list 语义改成 cluster-pair

### P2：只读路径清理与 `const __restrict__`

这是低风险的小优化项，适合作为补充项。

#### 目标

- 给编译器更明确的别名信息
- 争取更好的只读缓存和调度结果

#### 目标文件

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`

#### 执行内容

1. 在可确认无别名冲突的位置补充 `const __restrict__`
2. 清理 force-only 热路径中不必要的条件判断
3. 尽量减少热点代码中的无关分支

### P2：热点路径专门化

当前最热点的是标准 `force-only + direct Coulomb` 路径。

#### 目标

进一步减少热点中的模板分支干扰与控制流复杂度。

#### 方向

- 对最常用组合做更明确的专用实现
- 避免在热点中保留不必要的能量/virial 路径开销

### P3：参数表访问优化

这一项有可能带来局部收益，但当前不应先做。

#### 目标

优化 `Get_LJ_Type` 之后的 `A/B` 参数访问成本。

#### 可选方向

1. 合并 `(A, B)` 读取
2. 对低类型数体系尝试 constant memory
3. 对水体系等低类型场景做轻量专门化

#### 说明

这类优化只有在前面的 launch / writeback / staging 都做过之后，才值得投入。

### 暂不优先的方向

以下方向当前不作为主线：

1. 再次做 `sqrt/powf` 类算术重写
2. 直接做 AoS -> SoA 全局改造
3. 修改近邻表结构
4. 引入 cluster-pair 路径

原因：

- 第一项已验证收益不明显
- 第二项工程扰动过大
- 第三、四项超出当前分支边界

## 建议执行顺序

当前推荐的执行顺序如下：

1. Launch Configuration Sweep
2. 降低 `atomicAdd(frc[atom_j])` 压力
3. 标准 LJ kernel 的 `j` tile staging
4. `const __restrict__` 与只读路径清理
5. 参数表访问优化

## 基准测试纪律

每次优化都必须重新测量。没有新测出来的数据，就不能声称“这次优化有效”。

### 环境

```bash
source /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST/env_gpu.sh
export SPONGE_BIN=/home/ylj/SPONGE_GITHUB/SPONGE/.pixi/envs/dev-cuda12/bin/SPONGE
export GMX_BIN=/home/ylj/应用/gromacs-2026.1/install-cuda/bin/gmx
```

如果本次测试使用的是其他构建产物，必须在运行前显式更新 `SPONGE_BIN`。

### 编译命令

每次 benchmark 前使用当前源码重新编译 CUDA 二进制：

```bash
cd /home/ylj/SPONGE_GITHUB/SPONGE
pixi run -e dev-cuda12 compile
```

### Quick Regression Benchmark

用于每个 patch 后的快速回归：

```bash
cd /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST
python runner.py bench \
  --mode NVT \
  --engine sponge \
  --sponge-bin "$SPONGE_BIN" \
  --gpu-id 0 \
  --steps 20000 \
  --warmup 1 \
  --repeats 1 \
  --run-id lj_acc_quick_<tag>
```

必须记录：

- `elapsed_s`
- `steps_per_s`
- `ns_per_day`

### Acceptance Benchmark

用于每完成一个 P0 或 P1 阶段后的确认：

```bash
cd /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST
python runner.py bench \
  --mode NVT \
  --engine sponge \
  --sponge-bin "$SPONGE_BIN" \
  --gpu-id 0 \
  --steps 100000 \
  --warmup 1 \
  --repeats 3 \
  --run-id lj_acc_accept_<tag>
```

### 可选参考对照：GROMACS 1 CPU + 1 GPU

当需要判断 SPONGE 是否缩小与 GROMACS 单 CPU 配置的差距时，运行：

```bash
cd /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST
python runner.py bench \
  --mode NVT \
  --engine both \
  --sponge-bin "$SPONGE_BIN" \
  --gmx-bin "$GMX_BIN" \
  --gmx-ntomp 1 \
  --gpu-id 0 \
  --steps 20000 \
  --warmup 1 \
  --repeats 1 \
  --run-id lj_acc_vs_gmx1_<tag>
```

## Profiler 使用规则

只有当某次 patch 明显改变了 kernel 行为时，才重新做 profiler。

### Nsight Systems

用途：

- 观察 kernel 发射节奏是否退化
- 观察 GPU pipeline 是否被新的同步或空洞破坏

### Nsight Compute

重点关注以下指标是否改善：

- `SM Throughput`
- `Memory Throughput`
- `Achieved Occupancy`
- `L2 hit rate`
- stall breakdown

### profiler 结论规则

- 不允许仅根据单次 kernel 时间判断优化有效
- 必须结合 end-to-end benchmark 判断

## 每个 patch 必须记录的证据

每次 LJ 优化 patch 完成后，至少记录以下内容：

1. git commit hash
2. 改动范围
3. quick benchmark 的 `ns/day`
4. 与上一个基线相比的百分比变化
5. 数值行为是否仍可接受
6. 是否补充了 profiler 证据

推荐记录格式：

```markdown
| Commit | 改动内容 | Steps | ns/day | 相对变化 | 备注 |
|--------|----------|-------|--------|----------|------|
| abc123 | launch sweep: 32x8 | 20000 | 78.4 | +6.2% | quick benchmark |
```

## 成功标准

本分支只有在满足以下条件时，才算技术上成功：

1. 在相同公平参数下，SPONGE 的 `ns/day` 有稳定提升
2. 没有引入新的近邻表语义
3. 数值行为保持可接受
4. 每个阶段都有可复查的 benchmark 证据

