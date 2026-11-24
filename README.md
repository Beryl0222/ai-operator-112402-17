# 小窗相册增强的分块奇异值收缩与置信合成

项目资料使用可独立读写的 JSON。Node.js 22.23.4 下运行 `npm ci --offline` 准备环境，然后运行 `npm run inspect` 检查合成数据字段，运行 `npm test` 校验样本引用及格式。

## 输入契约

一个块按 frame×pixel 给亮度矩阵及有效遮罩。对同一 supportId 仅聚合仍共同有效的坐标，输入通过 alignQuality 阈值才可进矩阵。奇异值软阈值定义为 max(σ-λ,0)，零方向不加入重建。对重复非零奇异值以重建矩阵不变为准；近门限的不确定贡献能给区间，区级允许的附加误差为 profile.tolerance。块融合对各坐标的能够交付的值用 positions 的正权重加权，覆盖不足时输出原片及该事实。局部支持全空直接回退。

## 资料读取

样本位于 `inputs/session.json`，其中 `events` 是按到达顺序记下的操作，编号从零开始。frames 给出块 ID 和送达帧的下标区间，replace 给出块 ID、帧 ID 与新置信度，read 指半开像素索引范围。positions 每项为索引和融合权重。
