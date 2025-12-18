# 坐标变换
## 1. 输出:waypoint坐标系(dx,dy,dz) 
## 2. 坐标系定义
### waypoint坐标系：
- **x+**：正前方
- **y+**：正左方
- **z+**：正上方<br>
### habitat世界坐标系：
- **z-**：正前方
- **x-**：正上方
- **y+**：正上方<br>
---
## 3. 坐标变换矩阵:
$$
T=
    \begin{bmatrix}
    0 & -1 & 0\\
    0 & 0 & -1\\
    -1 & 0 & 0
    \end{bmatrix}
$$  
### habitat世界坐标与waypoint坐标转换关系：
- 
$$
\mathbf{p}_{\text{habitat}}
=
T_{\text{waypoint} \rightarrow \text{habitat}}
\;
\mathbf{p}_{\text{waypoint}}
$$

