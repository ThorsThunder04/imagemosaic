from random import randint, choice
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

def k_dist(pt1: tuple[int,...], pt2: tuple[int,...], k: int) -> float:
    
    square_sum = 0
    for i in range(k):
        square_sum += (pt1[i]-pt2[i])**2
    
    return square_sum**0.5

def closest_point_kdt(pt: tuple[int,...],
                      t: KdTree,
                      k: int) -> tuple[float, tuple[any, tuple[int,...]]]:
    """
    Find the closest point to a given point in a k-d Tree

    Paramaters
    ----------
    pt : (tuple[int,...])
        The target point we want to find the closest point to in `t`
    t : (KdTree)
        Root of the KdTree we are looking for a closest point for
    k : (int)
        The dimension of the k-d tree
    
    Returns
    -------
    (tuple[any, tuple[int,...]])
        The closest point to `pt`
    """
    # if there's closer points to the left
    if pt[t.axis] <= t.val[1][t.axis] and t.left is not None:
        cc_point = closest_point_kdt(pt, t.left, k)
        
        if t.right is not None:
            dist2 = k_dist(pt, t.right.val[1], k)
            if dist2 < cc_point[0]:
                cc_point2 = closest_point_kdt(pt, t.right, k)
                if cc_point2[0] < cc_point[0]:
                    cc_point = cc_point2
        
        return cc_point

    # if there's closer points to the right
    elif pt[t.axis] > t.val[1][t.axis] and t.right is not None:
        cc_point =  closest_point_kdt(pt, t.right, k)

        if t.left is not None:
            dist2 = k_dist(pt, t.left.val[1], k)
            if dist2 < cc_point[0]:
                cc_point2 = closest_point_kdt(pt, t.left, k)
                if cc_point2[0] < cc_point[0]:
                    cc_point = cc_point2
        
        return cc_point

    # otherwise it's a leaf / we can't go to a lower branch
    else:
        dist = k_dist(pt, t.val[1], k)
        return (dist, t.val)
    

if __name__ == "__main__":

    def prefix_disp(node):
        if node is not None:
            print(node.val)
            prefix_disp(node.left)
            prefix_disp(node.right)
    
    def merge_freq_dicts(dict1, dict2):
        for key,value in dict2.items():
            if key not in dict1:
                dict1[key] = 0
            
            dict1[key] += value
        
        return dict1
            
    
    def depth_freqs(t: KdTree, depth: int = 0):

        if t.is_leaf():
            return {depth: 1}
        else:
            right_depths = {}
            if t.right is not None:
                right_depths = depth_freqs(t.right, depth+1)
            
            left_depths = {}
            if t.left is not None:
                left_depths = depth_freqs(t.left, depth+1)
            
            return merge_freq_dicts(left_depths, right_depths)

    def closest_point_linear(pt: tuple[int,...],
                             points: list[tuple[any, tuple[int,...]]],
                             k: int) -> tuple[any, tuple[int,...]]:

        min_pt = (k_dist(pt, points[0][1], k), points[0])

        for i in range(1, len(points)):
            dist = k_dist(pt, points[i][1], k)
            if dist < min_pt[0]:
                min_pt = (dist, points[i])
        
        return min_pt
        
    
    # points = [("1", (1,2,3)), ("2", (4,3,3)), ("5", (23,44,12))]
    points = []
    for i in range(100):
        point = (randint(0, 1000), randint(0,1000), randint(0,1000))
        points.append( (str(i), point))
        

    t = build_kd_tree(points, 3)

    prefix_disp(t)
    print(t.make_string())
    # assert t.point_in_tree(("1", (1,2,3)))
    # assert not t.point_in_tree(("1", (5,2,3)))
    # assert t.point_in_tree(("2", (4,3,3)))
    print(depth_freqs(t))

    pt = ("", (432,12,774))
    print("Linear closest point to {} is {}".format(
        pt,
        closest_point_linear(pt[1], points, 3)
    ))
    print("k-d tree closest point to {} is {}".format(
        pt,
        closest_point_kdt(pt[1], t, 3)
    ))
