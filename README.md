# AbstraCMP

软件学报：基于多路径回溯的神经网络验证方法

[JOS 22] https://dx.doi.org/10.13328/j.cnki.jos.006585

[IJSI] https://doi.org/10.21655/ijsi.1673-7288.00281

本文提出多路径回溯的概念，现有的此类方法，如 DeepPoly，可看作仅使用单条回溯路径计算每个神经网络节点的上下界，是这一概念的特例。使用多条回溯路径可以有效改善这类方法的精度。

我们将多路径回溯方法与使用单条回溯路径的 DeepPoly 进行定量比较，结果表明多路径回溯方法能够获得明显的精度提升，而仅引入相对其他精华方法较小的额外时间代价。

This paper proposes the concept of back-propagation path for back-propagation methods. Under this concept, the existing back-propagation methods can be regarded as only using a single back-propagation path, which is a special case of the proposed concept.

The verification accuracy of back-propagation methods can be improved effectively by using multiple back-propagation paths. In this paper, a general back-propagation method using multiple propagation paths is formalized, and its reliability and verification accuracy are proved theoretically.
