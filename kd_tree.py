from random import sample
from math import ceil

class KdTree:
    
    def __init__(self,
                val: tuple[any, tuple[int,...]],
                k: int,
                axis: int,
                left = None,
                right = None):
        
        self.val = val
        self.left = left
        self.right = right
        self.dim = k
        self.axis = axis
        
    
    def is_leaf(self):
        """
        Returns wether or not the current node is a leaf or not (has no children)

        Returns
        -------
        (bool)
            `True` if both children are `None`, otherwise `False`
        """
        return self.left is None and self.right is None
    
    def point_in_tree(self, pt: tuple[any, tuple[int,...]]) -> bool:
        """
        Check if the given point `pt` is present in the current k-d tree (both the point's name and coordinates have to match)

        Paramaters
        ----------
        pt : (tuple[any, tuple[int,...]])
            The point we want to look for in the tree

        Returns
        -------
        (bool)
            `True` if the point is present in the tree, otherwise `False`
        """
        
        parser = self
        while not parser.is_leaf() and parser.val != pt:
            # if the axis value is less then the parser's value then check the left branch
            if pt[1][self.axis] <= parser.val[1][self.axis]:
                parser = parser.left
            else: # otherwise check the right banch
                parser = parser.right
        
        return parser.val == pt
    
        
    def make_string(self, depth: int = 0) -> str:
        if self.is_leaf():
            return "\t"*depth + str(self.val) + "\n"
        else:
            s = "\t"*depth + str(self.val) + "\n"
            if self.left is not None:
                s += self.left.make_string(depth + 1)
            else:
                s += "\t"*(depth+1) + "|\n"
            if self.right is not None:
                s += self.right.make_string(depth + 1)
            else:
                s += "\t"*(depth+1) + "|\n"
                
            return s
    


def build_kd_tree(points: list[tuple[any, tuple[int,...]]], 
                  k: int, 
                  _depth: int = 0
                  ) -> KdTree:
    """
    Builds a ~balanced k-d tree from given points in k dimensions

    Paramaters
    ----------
    points : (list[tuple[any,tuple[int,...]])
        The cloud of named points that will be in the kd-tree
    k : (int)
        The dimension of the points
    """
    
    axis = _depth % k
    if len(points) == 1:
        return KdTree(points[0], k, _depth % k)
    elif len(points) == 0:
        return None

    #TODO DO NOT SORT AT EACH RECURSIVE CALL! mabye sort everything ONCE at the very first call. The just pick out what we want. (like the 2PPP algo)
    points.sort(key = lambda x: x[1][axis])
    median = points[len(points)//2]


    t = KdTree(median, k, axis)

    # create left branch with all points inferior or equal to the median on the current axis
    left_partition = [point for point in points if point[1][axis] < median[1][axis]]
    t.left = build_kd_tree(left_partition, k, _depth+1)

    # create right branch with all points strictly superior to the median on the current axis
    right_partition = [point for point in points if point[1][axis] > median[1][axis]]
    t.right = build_kd_tree(right_partition, k, _depth+1)

    return t


if __name__ == "__main__":

    def prefix_disp(node):
        if node is not None:
            print(node.val)
            prefix_disp(node.left)
            prefix_disp(node.right)
        

    
    points = [("1", (1,2,3)), ("2", (4,3,3)), ("5", (23,44,12))]

    t = build_kd_tree(points, 3)

    prefix_disp(t)
    print(t.make_string())
    assert t.point_in_tree(("1", (1,2,3)))
    assert not t.point_in_tree(("1", (5,2,3)))
    assert t.point_in_tree(("2", (4,3,3)))
