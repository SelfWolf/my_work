def find_integer_in_array(matrix, num):
    """二维数组寻找是否包含num"""
    if not matrix:
        return False
    rows, cols = len(matrix), len(matrix[0])
    row, col = rows - 1, 0
    while row >=0 and col <= cols - 1:
        if matrix[row][col] == num:
            return True
        elif matrix[row][col] < num:
            col += 1
        else:
            row += 1
    return False


def replace_space_in_string(s):
    """将字符串中所有空格替换为20%"""
    return s.replace(' ', '20%')
    # import re
    # return re.sub('20%', ' ', s)


def print_links_reverse(links):
    """逆序打印里链表数值"""
    stack = []
    while links:
        stack.append(links.val)
        links = links.next
    while stack:
        print(stack.pop())
    """递归的打印链表内容"""
    if links:
        print_links_reverse(links)
        print(links.val)


def construct_tree(preorder, inorder):
    """
    基于前序和中序重构二叉树(类似知道中序和后序构建也行，只知道前序和后序，不是完全二叉树的前提下没法重构)
    前序的第一个数根节点，可以分割中序为左右两部分；左部分的长度也是前序左子树的所有节点
    递归的，前序第二个节点，是中序左部分的根节点
    :param preorder:
    :param inorder:
    :return:
    """
    if not preorder or not inorder:
        return None
    index = inorder.index(preorder[0])
    left = inorder[:index]
    right = inorder[index+1:]
    root = TreeNode(preorder[0])
    root.left = construct_tree(preorder[1:1+len(left)], left)
    root.right = construct_tree(preorder[-len(right):], right)
    return root


class MyQuene:
    """
    利用栈实现先进先出的队列：两个列表模拟
    1. 列表1当入栈，列表2当出栈；
    2. 如果列表2有值，直接list.pop操作；否则将列表1的值pop进列表2，然后在pop列表2
    """
    def __init__(self):
        self.stack1, self.stack2 = [], []

    def push(self, val):
        self.stack1.append(val)

    def pop(self):
        if self.stack2:
            return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop() if self.stack2 else None


def find_min_in_xuanzhuan_array(nums):
    """旋转数组中找最小值(找第k的值)"""
    if not nums:
        return False
    length = len(nums)
    left, right = 0, length - 1
    while nums[left] >= nums[right]:
        if right - left == 1:
            return nums[right]
        mid = (left + right) / 2
        if nums[left] == nums[mid] == nums[right]:
            return min(nums)
        if nums[left] <= nums[mid]:
            left = mid
        if nums[right] >= nums[mid]:
            right = mid
    return nums[0]


def num_of_one(n):
    """n的二进制表示中，1的个数"""
    ret = 0
    while n:
        ret += 1
        n = n & n-1
    return ret


def power(base, exponent):
    """考虑一个数的整数次方"""
    def equal_zero(num):
        if abs(num - 0.0) < 0.000001:
            return True

    def power_value(base, exponent):
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        ret = power_value(base, exponent >> 1)
        ret *= ret
        if exponent & 1 == 1:
            ret *= base
        return ret
    if equal_zero(base) and exponent < 0:
        return ZeroDivisionError
    ret = power_value(base, exponent)
    if exponent < 0:
        return 1.0 / ret
    else:
        return ret


def print_max_n(n):
    """打印从1到最大的n位数，Python对大整数进行了自动转换，不考虑溢出问题"""
    for i in range(1, 10 ** n):
        print(i)


def delete_node(link, node):
    """
    O(1)时间内删除链表节点
    相当于将node节点后的节点值覆盖node，然后删除node后的节点
    如果node没有后续节点 ，那么只能顺序查找到该节点
    """
    if link == node:
        del node
    if node.next is None:
        while link:
            if link.next == node:
                link.next = None
            link = link.next
    else:
        node.val = node.next.val
        n_node = node.next
        node.next = n_node.next
        del n_node


def reorder(nums):
    """
    将数组中奇数调整到偶数前面
    使用双指针，参考快速排序的思想
    """
    def is_even(num):
        return num % 2 == 0
    left, right = 0, len(nums) - 1
    while left < right:
        while not is_even(nums[left]):
            left += 1
        while is_even(nums[right]):
            right += 1
        if left < right:
            nums[left], nums[right] = nums[right], nums[left]


def last_kth(link, k):
    """求链表倒数第k个节点值；使用快慢指针"""
    if not link or k <= 0:
        return None
    move = link
    while move and k - 1 >= 0:
        move = move.next
        k -= 1
    while move:
        move = move.next
        link = link.next
    if k == 0:
        return link.val
    return None


def reverse_link(head):
    """链表翻转"""
    if not head or not head.next:
        return head
    then = head.next
    head.next = None
    last = then.next
    while then:
        then.next = head
        head = then
        then = last
        if then:
            last = then.next
    return head


def merge_link(head1, head2):
    """链表合并"""
    if not head1:
        return head2
    if not head2:
        return head1
    if head1.val <= head2.val:
        ret = head1
        ret.next = merge_link(head1.next, head2)
    else:
        ret = head2
        ret.next = merge_link(head1, head2.next)
    return ret


def sub_tree(tree1, tree2):
    """判断一个二叉树是否是另一个的子结构"""
    if tree1 and tree2:
        if tree1.val == tree2.val:
            return sub_tree(tree1.left, tree2.left) and sub_tree(tree1.right, tree2.right)
        else:
            return sub_tree(tree1.left, tree2.left) or sub_tree(tree1.right, tree2.right)
    if not tree2 and tree1:
        return False
    return True
