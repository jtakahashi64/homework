class Node:
    def __init__(self):
        self.l = None
        self.r = None
        self.data = None


# Binary Tree
# ノードの左側には、そのノードより小さい値を持つノードがある
# ノードの右側には、そのノードより大きい値を持つノードがある
class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = Node()
            self.root.data = data
        else:
            self._insert_recursive(data, self.root)

    def _insert_recursive(self, data, node):
        if data < node.data:
            if node.l is None:
                node.l = Node()
                node.l.data = data
            else:
                self._insert_recursive(data, node.l)
        else:
            if node.r is None:
                node.r = Node()
                node.r.data = data
            else:
                self._insert_recursive(data, node.r)

    def delete(self, value):
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        if node is None:
            return node
        if value < node.value:
            node.l = self._delete_recursive(node.l, value)
        if value > node.value:
            node.r = self._delete_recursive(node.r, value)
        if value == node.value:
            # Node with only one child or no child
            if node.l is None:
                temp = node.r
                node = None
                return temp
            if node.r is None:
                temp = node.l
                node = None
                return temp

            # Node with two children: Get the inorder successor (smallest in the right subtree)
            # In-order Successor
            # 削除後 右 均衡が維持される
            # 削除後 左 均衡が維持される
            temp = self._min_value_node(node.r)
            node.value = temp.value
            # Delete the inorder successor
            node.r = self._delete_recursive(node.r, temp.value)
        return node

    def _min_value_node(self, node):
        current = node
        while current.l is not None:
            current = current.l
        return current

    def contains(self, value):
        return self._contains_recursive(self.root, value)

    def _contains_recursive(self, node, value):
        if node is None:
            return False
        if node.data == value:
            return True
        if value < node.data:
            return self._contains_recursive(node.l, value)
        else:
            return self._contains_recursive(node.r, value)

    def height(self):
        return self._height_recursive(self.root)

    def _height_recursive(self, node):
        if node is None:
            return -1
        else:
            l_height = self._height_recursive(node.l)
            r_height = self._height_recursive(node.r)
            return max(l_height, r_height) + 1

    # Traversal Techniques
    # - inorder, preorder, postorder
    # - inorder
    # ref: https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/
    def print_in_order(self):
        self._print_in_order_recursive(self.root)

    def _print_in_order_recursive(self, node):
        if node is not None:
            self._print_in_order_recursive(node.l)
            print(node.value)
            self._print_in_order_recursive(node.r)
