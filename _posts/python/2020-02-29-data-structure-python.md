---
layout: post
title: '数据结构与算法(Python)'
date: 2020-02-29
author: 郑之杰
cover: 'https://pic.downk.cc/item/5e89be33504f4bcb040382b2.jpg'
tags: Python
---

> Data Structure and Algorithm with Python.

**数据结构(Data Structure)**描述了数据的存储方式。**Python**的内置数据结构包括列表(**list**)、元组(**tuple**)、字典(**dictionary**)、集合(**set**)等。我们需要自己定义实现**python**内没有定义的扩展数据结构。

**抽象数据类型(Abstract Data Type)**是指把数据类型和数据类型上的运算封装在一起。最常用的数据运算：插入、删除、修改、查找、排序。


### ⚪ 目录：

1. [时间复杂度](https://0809zheng.github.io/2020/04/01/data-structure-python.html#1-%E6%97%B6%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6)
2. [线性表](https://0809zheng.github.io/2020/04/01/data-structure-python.html#2-%E7%BA%BF%E6%80%A7%E8%A1%A8)：顺序表、链表(单向链表, 双向链表, 单向循环链表)、栈、队列(双端队列)
5. [排序](https://0809zheng.github.io/2020/04/01/data-structure-python.html#3-%E6%8E%92%E5%BA%8F-sorting)：冒泡排序、选择排序、插入排序、希尔排序、快速排序、归并排序、堆排序、计数排序、桶排序、基数排序
6. [搜索](https://0809zheng.github.io/2020/04/01/data-structure-python.html#4-%E6%90%9C%E7%B4%A2)：二分查找
7. [树](https://0809zheng.github.io/2020/04/01/data-structure-python.html#5-%E6%A0%91)：二叉树、堆、四叉树与八叉树

# 1. 时间复杂度
衡量一个算法的优劣，单看运行时间是不可靠的；程序的运行离不开计算机环境（包括硬件与操作系统）。

假设计算机执行算法时每一个基本操作的时间是单位时间；对于不同的机器环境，单位时间是不同的，但是对于算法执行多少个基本操作在规模数量级上是相同的。

**大O记法（big-O）**：对于单调整数函数f，如果存在一个整数函数g和正实数c，使得对充分大的n总有$f(n)≤cg(n)$，则称函数g是函数f的一个渐进函数，记为$f(n)=O(g(n))$。

**时间复杂度**：假设存在函数g，使得算法A处理规模为n的问题所用的时间为$T(n)=O(g(n))$，则称$O(g(n))$为算法A的渐进时间复杂度，简称时间复杂度，记为$T(n)$。

- **最优时间复杂度**：算法完成工作最少需要多少基本操作。
- **平均时间复杂度**：算法完成工作平均需要多少基本操作。
- **最坏时间复杂度**：算法完成工作最多需要多少基本操作。

时间复杂度的基本计算规则：
1. 基本操作，只有常数项，时间复杂度为$O(1)$；
2. 顺序结构，时间复杂度按加法进行计算；
3. 循环结构，时间复杂度按乘法进行计算；
4. 分支结构，时间复杂度取最大值；
5. 判断算法效率，只需要关注最高次项，其他次要项和常数项可以忽略；
6. 没有特殊说明时，时间复杂度一般指最坏时间复杂度。

常见时间复杂度的关系：

$$ O(1) < O(\log(n)) < O(n) < O(n\log(n)) < O(n^2) < O(n^3) < O(2^n) < O(n!) < O(n^n) $$

# 2. 线性表
**线性表**是数据元素的集合，它记录着元素之间的顺序关系。根据线性表的实际存储方式，分为两种模型：

- **顺序表**，将元素顺序地存放在一块连续的存储区里，元素间的顺序关系由它们的存储顺序自然表示。
- **链表**，将元素存放在通过链接构造起来的一系列存储块中。

## （1）顺序表
### ①顺序表的基本形式
![](https://pic.downk.cc/item/5e844f1e504f4bcb040386a2.png)

上图a)是顺序表的**基本形式**，数据元素按顺序存储，每个元素所占的存储单元大小固定相同。

元素的下标是其逻辑地址，而元素存储的物理地址（实际内存地址）可以通过存储区的起始地址$Loc(e_0)$加上逻辑地址（第i个元素）与存储单元大小（c）的乘积计算而得，即：

$$ Loc(e_i)=Loc(e_0)+c\times i $$

访问指定元素时无需从头遍历，通过计算便可获得对应地址，其时间复杂度为$O(1)$。

如果元素的大小不统一，则须采用图b)的**元素外置的形式**(也被称为对实际数据的索引)，将实际数据元素另行存储，而顺序表中各单元位置保存对应元素的地址信息（即链接）。由于每个链接所需的存储量相同，通过上述公式，可以计算出元素链接的存储位置，而后顺着链接找到实际存储的数据元素。注意，图b)中的c不再是数据元素的大小，而是存储一个链接地址所需的存储量，这个量通常很小。

一个顺序表的完整信息包括两部分，一部分是表中的元素集合，另一部分是为实现正确操作而需记录的信息，即有关表的整体情况的信息，这部分信息主要包括元素存储区的容量和当前表中已有的元素个数两项。

### ②顺序表的实现方式
![](https://pic.downk.cc/item/5e8451db504f4bcb040546ab.jpg)

图a)为**一体式结构**，存储表信息的单元与元素存储区以连续的方式安排在一块存储区里，两部分数据的整体形成一个完整的顺序表对象。

一体式结构整体性强，易于管理。但是由于数据元素存储区域是表对象的一部分，顺序表创建后，元素存储区就固定了。

图b)为**分离式结构**，表对象里只保存与整个表有关的信息（即容量和元素个数），实际数据元素存放在另一个独立的元素存储区里，通过链接与基本表对象关联。

### ③顺序表的操作

Ⅰ. 元素存储区替换

一体式结构由于顺序表信息区与数据区连续存储在一起，所以若想更换数据区，则只能整体搬迁，即整个顺序表对象（指存储顺序表的结构信息的区域）改变了。

分离式结构若想更换数据区，只需将表信息区中的数据区链接地址更新即可，而该顺序表对象不变。

Ⅱ. 元素存储区扩充

采用分离式结构的顺序表，若将数据区更换为存储空间更大的区域，则可以在不改变表对象的前提下对其数据存储区进行了扩充，所有使用这个表的地方都不必修改。只要程序的运行环境（计算机系统）还有空闲存储，这种表结构就不会因为满了而导致操作无法进行。人们把采用这种技术实现的顺序表称为**动态顺序表**，因为其容量可以在使用中动态变化。

扩充的两种策略：
- 每次扩充增加固定数目的存储位置，如每次扩充增加10个元素位置，这种策略可称为线性增长。特点：节省空间，但是扩充操作频繁，操作次数多。
- 每次扩充容量加倍，如每次扩充增加一倍存储空间。特点：减少了扩充操作的执行次数，但可能会浪费空间资源。以空间换时间，推荐的方式。

Ⅲ. 增加元素
![](https://pic.downk.cc/item/5e8453fc504f4bcb04067b65.jpg)

a. 尾端加入元素，时间复杂度为O(1);

b. 非保序的加入元素（不常见），时间复杂度为O(1);

c. 保序的元素加入，时间复杂度为O(n)。

Ⅳ. 删除元素
![](https://pic.downk.cc/item/5e845416504f4bcb04068d73.jpg)

a. 删除表尾元素，时间复杂度为O(1);

b. 非保序的元素删除（不常见），时间复杂度为O(1);

c. 保序的元素删除，时间复杂度为O(n)。

### ④python中的顺序表

**Python**中的**list**和**tuple**两种类型采用了顺序表的实现技术，具有前面讨论的顺序表的所有性质。

**tuple**是不可变类型，即不变的顺序表，因此不支持改变其内部状态的任何操作，而其他方面，则与**list**的性质类似。

**Python**标准类型**list**是一种采用**分离式**技术实现的**动态顺序表**，可以加入和删除元素，并在各种操作中维持已有元素的顺序（即保序），而且还具有以下行为特征：

- 基于下标（位置）的高效元素访问和更新，时间复杂度应该是**O(1)**；为满足该特征，应该采用顺序表技术，表中元素保存在一块连续的存储区中。
- 允许任意加入元素，而且在不断加入元素的过程中，表对象的标识（函数id得到的值）不变。为满足该特征，就必须能更换元素存储区，并且为保证更换存储区时**list**对象的标识id不变，只能采用分离式实现技术。

在Python的官方实现中，**list**实现采用了如下的策略：在建立空表（或者很小的表）时，系统分配一块能容纳8个元素的存储区；在执行插入操作（**insert**或**append**）时，如果元素存储区满就换一块4倍大的存储区。但如果此时的表已经很大（目前的阀值为**50000**），则改变策略，采用加一倍的方法。引入这种改变策略的方式，是为了避免出现过多空闲的存储位置。

![](https://pic.downk.cc/item/5e845639504f4bcb0407db33.jpg)

## （2）链表 Link List
顺序表的构建需要预先知道数据大小来申请连续的存储空间，而在进行扩充时又需要进行数据的搬迁，所以使用起来并不是很灵活。

链表结构可以充分利用计算机内存空间，实现灵活的内存动态管理。

**链表（Linked list）**是一种常见的基础数据结构，是一种线性表，但是不像顺序表一样连续存储数据，而是在每一个节点（数据存储单元）里存放下一个节点的位置信息（即地址）。
![](https://pic.downk.cc/item/5e847674504f4bcb041d2e12.jpg)

### ⚪ 单向链表 Single Link List
**单向链表**也叫单链表，是链表中最简单的一种形式，它的每个节点包含两个域，一个**信息域（元素域）**和一个**链接域**。这个链接指向链表中的下一个节点，而最后一个节点的链接域则指向一个空值。
![](https://pic.downk.cc/item/5e84771f504f4bcb041d8977.jpg)

- 元素域**elem**用来存放具体的数据;
- 链接域**next**用来存放下一个节点的位置（**python**中的标识，在**python**中为变量名，代表对象的地址）;
- 变量p指向链表的头节点（首节点）的位置，从p出发能找到表中的任意节点。

单链表的操作：
- is_empty() 链表是否为空
- length() 链表长度
- travel() 遍历整个链表
- add(item) 链表头部添加元素
- append(item) 链表尾部添加元素
- insert(pos, item) 指定位置添加元素
- remove(item) 删除节点
- search(item) 查找节点是否存在

节点的实现：
```python
class Node(object):
    """单链表的节点"""
    def __init__(self, elem):
        # elem存放数据元素
        self.elem = elem
        # next是下一个节点的标识
        self.next = None
```

单向链表的实现：
```python
class SingleLinkList(object):
    def __init__(self, node = None):
        self.__head = node

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head is None  # 和None比较推荐用 is 而不是 ==
		
    def length(self):
        """链表长度"""
        cursor = self.__head
        count = 0
        while cursor is not None:
            count += 1
            cursor = cursor.next
        return count
		
    def travel(self):
        """遍历链表"""
        cursor = self.__head
        while cursor is not None:
            print(cursor.elem, end = ' ')
            cursor = cursor.next
        print('')
```

尾部添加元素:
```python
    def append(self, elem):
        """尾部添加元素"""
        node = Node(elem)
        # 先判断链表是否为空，若是空链表，则将__head指向新节点
        if self.is_empty():
            self.__head = node
        # 若不为空，则找到尾部，将尾节点的next指向新节点
        else:
            cursor = self.__head
            while cursor.next is not None:
                cursor = cursor.next
            cursor.next = node
```

头部添加元素:
![](https://pic.downk.cc/item/5e84931b504f4bcb0431a02d.jpg)
```python
    def add(self, elem):
        """头部添加元素"""
        node = Node(elem)
        node.next = self.__head
        self.__head = node
```

指定位置添加元素:
![](https://pic.downk.cc/item/5e849430504f4bcb04328d3f.jpg)
```python
    def insert(self, pos, elem):
        """指定位置添加元素"""
        # 若指定位置pos为第一个元素之前，则执行头部插入
        if pos <= 0:
            self.add(elem)
        # 若指定位置超过链表尾部，则执行尾部插入
        elif pos > (self.length()-1):
            self.append(elem)
        else:
            cursor = self.__head
            count = 0
            while count < pos - 1:
                count += 1
                cursor = cursor.next
            node = Node(elem)
            node.next = cursor.next
            cursor.next = node
```

删除节点:
![](https://pic.downk.cc/item/5e849bf3504f4bcb04387f76.jpg)
```python
    def remove(self, elem):
        """删除节点"""
        prior = None
        cursor = self.__head
        while cursor is not None:
            if cursor.elem == elem:
                # 如果第一个就是删除的节点
                if cursor == self.__head:
                    self.__head = cursor.next
                else:
                    prior.next = cursor.next
                break
            prior = cursor
            cursor = cursor.next
```

查找节点是否存在:
```python
    def search(self, elem):
        """查找节点是否存在，并返回True或者False"""
        cursor = self.__head
        while cursor is not None:
            if cursor.elem == elem:
                return True
            cursor = cursor.next
        return False
```

链表与顺序表的对比:

链表失去了顺序表随机读取的优点，同时链表由于增加了结点的指针域，空间开销比较大，但对存储空间的使用要相对灵活。

链表与顺序表的各种操作复杂度如下所示：
![](https://pic.downk.cc/item/5e84a1d6504f4bcb043cea72.jpg)

注意虽然表面看起来复杂度都是 **O(n)**，但是链表和顺序表在插入和删除时进行的是完全不同的操作。链表的主要耗时操作是遍历查找，删除和插入操作本身的复杂度是**O(1)**。顺序表查找很快，主要耗时的操作是拷贝覆盖。因为除了目标元素在尾部的特殊情况，顺序表进行插入和删除时需要对操作点之后的所有元素进行前后移位操作，只能通过拷贝和覆盖的方法进行。

### ⚪ 双向链表 Double Link List
一种更复杂的链表是**“双向链表”**或“双面链表”。每个节点有两个链接：一个指向前一个节点（前驱区），当此节点为第一个节点时，指向空值；而另一个指向下一个节点（后继区），当此节点为最后一个节点时，指向空值。
![](https://pic.downk.cc/item/5e85513a504f4bcb04a824c6.jpg)

双链表的操作：
- is_empty() 链表是否为空
- length() 链表长度
- travel() 遍历整个链表
- add(item) 链表头部添加元素
- append(item) 链表尾部添加元素
- insert(pos, item) 指定位置添加元素
- remove(item) 删除节点
- search(item) 查找节点是否存在

节点的实现：
```python
class Node(object):
    """双链表的节点"""
    def __init__(self, elem):
        # elem存放数据元素
        self.elem = elem
        # next是下一个节点的标识
        self.next = None
        # prev是上一个节点的标识
        self.prev = None
```

双向链表的实现，其中**is_empty、length、travel、search**操作继承自单链表**SingleLinkList**：
```python
class DoubleLinkList(SingleLinkList):
    def __init__(self, node=None):
        self.__head = node
```

尾部添加元素:
```python
    def append(self, elem):
        """尾部添加元素"""
        node = Node(elem)
        # 先判断链表是否为空，若是空链表，则将__head指向新节点
        if self.is_empty():
            self.__head = node
        # 若不为空，则找到尾部，将尾节点的next指向新节点
        else:
            cursor = self.__head
            while cursor.next is not None:
                cursor = cursor.next
            cursor.next = node
            node.prev = cursor
```

头部添加元素:
```python
    def add(self, elem):
        """头部添加元素"""
        node = Node(elem)
        # 先判断链表是否为空，若是空链表，则将__head指向新节点
        if self.is_empty():
            self.__head = node
        # 若不为空
        else:
            node.next = self.__head
            node.next.prev = node
            self.__head = node
```

指定位置添加元素:
![](https://pic.downk.cc/item/5e855263504f4bcb04a8aa35.jpg)
```python
    def insert(self, pos, elem):
        """指定位置添加元素"""
        # 若指定位置pos为第一个元素之前，则执行头部插入
        if pos <= 0:
            self.add(elem)
        # 若指定位置超过链表尾部，则执行尾部插入
        elif pos > (self.length()-1):
            self.append(elem)
        else:
            cursor = self.__head
            count = 0
            while count < pos:
                count += 1
                cursor = cursor.next
            node = Node(elem)
            node.next = cursor
            node.prev = cursor.prev
            cursor.prev.next = node
            cursor.prev = node
```

删除节点:
![](https://pic.downk.cc/item/5e85526f504f4bcb04a8b09b.jpg)
```python
    def remove(self, elem):
        """删除节点"""
        cursor = self.__head
        while cursor is not None:
            if cursor.elem == elem:
                # 如果第一个就是删除的节点
                if cursor == self.__head:
                    self.__head = cursor.next
                    # 如果链表不是只有这一个节点
                    if cursor.next:
                        cursor.next.prev = None
                else:
                    cursor.prev.next = cursor.next
                    # 如果不是尾节点
                    if cursor.next:
                        cursor.next.prev = cursor.prev
                break
            cursor = cursor.next
```

### ⚪ 单向循环链表 Single Circle Link List
单链表的一个变形是**单向循环链表**，链表中最后一个节点的next域不再为None，而是指向链表的头节点。

![](https://pic.downk.cc/item/5e855c50504f4bcb04ada3a8.jpg)

单向循环链表的操作：
- is_empty() 链表是否为空
- length() 链表长度
- travel() 遍历整个链表
- add(item) 链表头部添加元素
- append(item) 链表尾部添加元素
- insert(pos, item) 指定位置添加元素
- remove(item) 删除节点
- search(item) 查找节点是否存在

单向循环链表的实现，其中节点同单链表，**is_empty、insert**操作继承自单链表**SingleLinkList**：
```python
class SingleCircleLinkList(SingleLinkList):
    def __init__(self, node = None):
        self.__head = node
        if node:
            node.next = node
		
    def length(self):
        """链表长度"""
        if is_empty():
            return 0
        cursor = self.__head
        count = 1
        while cursor.next != self.__head:
            count += 1
            cursor = cursor.next
        return count
		
    def travel(self):
        """遍历链表"""
        if self.__head is None:
            return
        cursor = self.__head
        while cursor.next != self.__head:
            print(cursor.elem, end = ' ')
            cursor = cursor.next
        print(cursor.elem)
```

尾部添加元素:
```python
    def append(self, elem):
        """尾部添加元素"""
        node = Node(elem)
        # 先判断链表是否为空，若是空链表，则将__head指向新节点
        if self.is_empty():
            self.__head = node
            node.next = node
        # 若不为空，则找到尾部，将尾节点的next指向新节点
        else:
            cursor = self.__head
            while cursor.next != self.__head:
                cursor = cursor.next
            cursor.next = node
            node.next = self.__head
```

头部添加元素:
```python
    def add(self, elem):
        """头部添加元素"""
        node = Node(elem)
        # 先判断链表是否为空，若是空链表，则将__head指向新节点
        if self.is_empty():
            self.__head = node
            node.next = node
        # 若不为空
        else:
            cursor = self.__head
            while cursor.next != self.__head:
                cursor = cursor.next
            cursor.next = node
            node.next = self.__head
            self.__head = node
```

删除节点:
```python
    def remove(self, elem):
        """删除节点"""
        if self.__head is None:
            return
        prior = None
        cursor = self.__head
        while cursor.next != self.__head:
            if cursor.elem == elem:
                # 如果第一个就是删除的节点
                if cursor == self.__head:
                    # 如果链表只有这一个节点
                    if cursor.next == self.__head:
                        self.__head = None
                    else:
                        prior = self.__head
                        while prior.next != self.__head:
                            prior = prior.next
                        prior.next = cursor.next
                        self.__head = cursor.next
                # 如果是中间节点
                else:
                    prior.next = cursor.next
                return  # 由于break退出循环不能退出函数，此处用return
            prior = cursor
            cursor = cursor.next
        # 如果是尾节点
        if cursor.elem == elem:
            # 如果链表只有这一个节点
            if prior is None:
                self.__head = None
            else:
                prior.next = self.__head
```

查找节点是否存在:
```python
    def search(self, elem):
        """查找节点是否存在，并返回True或者False"""
        if self.__head is None:
            return False
        cursor = self.__head
        while cursor.next != self.__head:
            if cursor.elem == elem:
                return True
            cursor = cursor.next
        if cursor.elem == elem:
            return True
        return False
```
## （3） 栈 Stack
**栈（stack）**，有些地方称为堆栈，是一种容器，可存入数据元素、访问元素、删除元素，它的特点在于只能允许在容器的一端（称为栈顶端指标，**top**）进行加入数据（**push**）和输出数据（**pop**）的运算。没有了位置概念，保证任何时候可以访问、删除的元素都是此前最后存入的那个元素，确定了一种默认的访问顺序。

由于栈数据结构只允许在一端进行操作，因而按照**后进先出（LIFO, Last In First Out）**的原理运作。
![](https://pic.imgdb.cn/item/6402a360f144a01007dd9ba4.jpg)

栈可以用顺序表实现，也可以用链表实现。用**python**内置的**list**实现如下：

栈的操作：
- Stack() 创建一个新的空栈
- push(item) 添加一个新的元素item到栈顶
- pop() 弹出栈顶元素
- peek() 返回栈顶元素
- is_empty() 判断栈是否为空
- size() 返回栈的元素个数

```python
class Stack(object):
    def __init__(self)：
        self.__list = []
    
    def push(self, item):
        self.__list.append(item)
		
    def pop(self):
        return self.__list.pop()
		
    def peek(self):
        if not self.__list:
            return None
        else:
            return self.__list[-1]
			
    def is_empty(self):
        return self.__list == []
		
    def size(self):
        return len(self.__list)
```

## （4） 队列 Queue
**队列（queue）**是只允许在一端进行插入操作，而在另一端进行删除操作的线性表。

队列是一种**先进先出的（First In First Out, FIFO）**的线性表。允许插入的一端为队尾，允许删除的一端为队头。队列不允许在中间部位进行操。假设队列是**q=（a1，a2，……，an）**，那么**a1**就是队头元素，而**an**是队尾元素。这样我们就可以删除时，总是从**a1**开始，而插入时，总是在队列最后。这也比较符合我们通常生活中的习惯，排在第一个的优先出列，最后来的当然排在队伍最后。
![](https://pic.downk.cc/item/5e85c2a0504f4bcb04fcf3a0.jpg)

### ⚪ 队列的实现
同栈一样，队列也可以用顺序表或者链表实现。用**python**内置的**list**实现如下：

队列的操作：
- Queue() 创建一个空的队列
- enqueue(item) 往队列中添加一个item元素
- dequeue() 从队列头部删除一个元素
- is_empty() 判断一个队列是否为空
- size() 返回队列的大小

```python
class Queue(object):
    def __init__(self):
        self.__list = []
		
    def enqueue(self, item):
        self.__list.append(item)
		
    def dequeue(self, item):
        return self.__list.pop(0)
		
    def is_empty(self):
        return self.__list == []
		
    def size(self):
        return len(self.__list)
```

### ⚪ 双端队列
**双端队列（deque，全名double-ended queue）**，是一种具有队列和栈的性质的数据结构。

双端队列中的元素可以从两端弹出，其限定插入和删除操作在表的两端进行。双端队列可以在队列任意一端入队和出队。
![](https://pic.downk.cc/item/5e85e6e9504f4bcb049910e2.jpg)

双端队列的操作：
- Deque() 创建一个空的双端队列
- add_front(item) 从队头加入一个item元素
- add_rear(item) 从队尾加入一个item元素
- remove_front() 从队头删除一个item元素
- remove_rear() 从队尾删除一个item元素
- is_empty() 判断双端队列是否为空
- size() 返回队列的大小

```python
class Deque(object):
    def __init__(self):
        self.__list = []
		
    def add_front(self, item):
        self.__list.insert(0, item)
		
    def add_rear(self, item):
        self.__list.append(item)
		
    def remove_front(self, item):
        return self.__list.pop(0)
		
    def remove_rear(self, item):
        return self.__list.pop()
		
    def is_empty(self):
        return self.__list == []
		
    def size(self):
        return len(self.__list)
```

# 3. 排序 Sorting
**排序(sorting)**是一种能将一串数据依照特定顺序进行排列的一种算法。

排序算法的**稳定性**：稳定排序算法会让原本有相等键值的纪录维持相对次序。也就是如果一个排序算法是稳定的，当有两个相等键值的纪录R和S，且在原本的列表中R出现在S之前，在排序过的列表中R也将会是在S之前。

当相等的元素是无法分辨的，比如像是整数，稳定性并不是一个问题。然而，假设以下的数对将要以他们的第一个数字来排序。

$$ (4, 1)  (3, 1)  (3, 7)  (5, 6) $$

在这个状况下，有可能产生两种不同的结果，一个是让相等键值的纪录维持相对的次序，而另外一个则没有：

$$ (3, 1)  (3, 7)  (4, 1)  (5, 6)  （维持次序） $$

$$ (3, 7)  (3, 1)  (4, 1)  (5, 6)  （次序被改变） $$

不稳定排序算法可能会在相等的键值中改变纪录的相对次序，但是稳定排序算法从来不会如此。不稳定排序算法可以被特别地实现为稳定。作这件事情的一个方式是人工扩充键值的比较，如此在其他方面相同键值的两个对象间之比较，（比如上面的比较中加入第二个标准：第二个键值的大小）就会被决定使用在原先数据次序中的条目，当作一个同分决赛。然而，要记住这种次序通常牵涉到额外的空间负担。

常见排序算法效率比较：
![](https://pic.imgdb.cn/item/640a9b9af144a010071f1a36.jpg)

⭐扩展阅读：[十大经典排序算法](https://www.runoob.com/w3cnote/ten-sorting-algorithm.html)、[常见排序算法的动图演示](https://visualgo.net/zh/sorting?slide=1-1)

## （1）冒泡排序
**冒泡排序（Bubble Sort）**是一种简单的排序算法。它重复地遍历要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。遍历数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。

冒泡排序算法的运作如下：
- 比较相邻的元素。如果第一个比第二个大（升序），就交换他们两个;
- 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数;
- 针对所有的元素重复以上的步骤，除了最后一个;
- 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。

![](https://pic.imgdb.cn/item/6402b3f7f144a01007f6455a.gif)

实现：

```python
def bubble_sort(alist):
    n = len(alist)
    for j in range(n-1):
        count = 0
        for i in range(n-1-j):
            if alist[i] > alist[i+1]:
                alist[i], alist[i+1] = alist[i+1], alist[i]
                count += 1
        # 遍历一次发现没有任何可以交换的元素
        if count == 0:
            return
```

时间复杂度：
- 最优时间复杂度：$O(n)$ （表示遍历一次发现没有任何可以交换的元素，排序结束。）
- 最坏时间复杂度：$O(n^2)$
- 稳定性：稳定

## （2）选择排序
**选择排序（Selection Sort）**是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

选择排序的主要优点与数据移动有关。如果某个元素位于正确的最终位置上，则它不会被移动。选择排序每次交换一对元素，它们当中至少有一个将被移到其最终位置上，因此对n个元素的表进行排序总共进行至多n-1次交换。在所有的完全依靠交换去移动元素的排序方法中，选择排序属于非常好的一种。

![](https://pic.imgdb.cn/item/6402f8fff144a0100769891d.gif)

实现：

```python
def select_sort(alist):
    n = len(alist)
    for j in range(n-1):
        min_index = j
        for i in range(j+1, n):
            if alist[min_index] > alist[i]:
                min_index = i
        alist[j], alist[min_index] = alist[min_index], alist[j]
```

时间复杂度：
- 最优时间复杂度：$O(n^2)$
- 最坏时间复杂度：$O(n^2)$
- 稳定性：不稳定（考虑升序每次选择最大的情况）

## （3）插入排序
**插入排序（Insertion Sort）**是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。

![](https://pic.imgdb.cn/item/6402f927f144a0100769c4a3.gif)

实现：

```python
def insert_sort(alist):
    n = len(alist)
    for i in range(1, n):
        while i > 0:
            if alist[i] < alist[i-1]:
                alist[i], alist[i-1] = alist[i-1], alist[i]
                i -= 1
            else:
                break
```

时间复杂度：
- 最优时间复杂度：$O(n)$ （升序排列，序列已经处于升序状态）
- 最坏时间复杂度：$O(n^2)$
- 稳定性：稳定

## （4）希尔排序
**希尔排序(Shell Sort)**是插入排序的一种。也称缩小增量排序，是直接插入排序算法的一种更高效的改进版本。希尔排序是非稳定排序算法。该方法因**DL. Shell**于**1959**年提出而得名。 希尔排序是把记录按下标的一定增量分组，对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的关键词越来越多，当增量减至1时，整个文件恰被分成一组，算法便终止。

![](https://pic.imgdb.cn/item/6402fc02f144a010076e217c.gif)

实现：

```python
def shell_sort(alist):
    n = len(alist)
    gap = n//2
    while gap > 0:
        for i in range(gap, n):
            while i >= gap:
                if alist[i] < alist[i-gap]:
                    alist[i], alist[i-gap] = alist[i-gap], alist[i]
                    i -= gap
                else:
                    break
        gap //= 2
```

时间复杂度：
- 最优时间复杂度：根据步长序列的不同而不同
- 最坏时间复杂度：$O(n^2)$
- 稳定性：不稳定

## （5）快速排序
**快速排序（Quicksort）**，又称划分交换排序（**partition-exchange sort**），通过一趟排序将要排序的数据分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。

步骤为：
1. 从数列中挑出一个元素，称为"**基准**"或"**枢**"（**pivot**）；
2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区结束之后，该基准就处于数列的中间位置。这个称为**分区**（**partition**）操作；
3. 递归地（**recursive**）把小于基准值元素的子数列和大于基准值元素的子数列排序。

递归的最底部情形，是数列的大小是零或一，也就是永远都已经被排序好了。虽然一直递归下去，但是这个算法总会结束，因为在每次的迭代（**iteration**）中，它至少会把一个元素摆到它最后的位置去。

![](https://pic.imgdb.cn/item/64030166f144a0100775c09d.gif)

```python
def quick_sort(alist, start, end):
    if start >= end:
        return
		
    pivot = alist[start]
    low = start
    high = end
	
    while low < high:
        while low < high and alist[high] >= pivot:
            high -= 1
        alist[low] = alist[high]
		
        while low < high and alist[low] < pivot:
            low += 1
        alist[high] = alist[low]
		
    alist[low] = pivot
	
    quick_sort(alist, start, low-1)  # 注意不能用alist的切片，因为这是一个新列表！
    quick_sort(alist, low+1, end)
```

时间复杂度：
- 最优时间复杂度：$O(n\log(n))$
- 最坏时间复杂度：$O(n^2)$
- 稳定性：不稳定

从一开始快速排序平均需要花费$O(n\log(n))$时间的描述并不明显。但是不难观察到的是分区运算，数组的元素都会在每次循环中走访过一次，使用$O(n)$的时间。在使用结合（**concatenation**）的版本中，这项运算也是$O(n)$。

在最好的情况，每次我们运行一次分区，我们会把一个数列分为两个几近相等的片段。这个意思就是每次递归调用处理一半大小的数列。因此，在到达大小为一的数列前，我们只要作$\log(n)$次嵌套的调用。这个意思就是调用树的深度是$O(\log(n))$。但是在同一层次结构的两个程序调用中，不会处理到原来数列的相同部分；因此，程序调用的每一层次结构总共全部仅需要$O(n)$的时间（每个调用有某些共同的额外耗费，但是因为在每一层次结构仅仅只有$O(n)$个调用，这些被归纳在$O(n)$系数中）。结果是这个算法仅需使用$O(n\log(n))$时间。

### ⚪ 优化快速排序

快速排序的平均时间复杂度是$O(n\log(n))$，但是最坏时间复杂度为$O(n^2)$。下面介绍一些改进快排的技巧。

在快速排序选择**pivot**时，默认为从前向后（或从后向前）选择。可以考虑从当前区间中随机地选择一个元素作为**pivot**，并交换到队首（或队尾）：

```python
#    pivot = alist[start]
    """
    优化pivot选择方法: 随机选择[start, end]中的任意元素
                      并交换到start位置处
    """
    idx = random.choice(range(start, end+1))
    alist[start], alist[idx] = alist[idx], alist[start]
    pivot = alist[start]
```

在快排中，把**pivot**安放在合适的位置后，若**pivot**前后两侧存在与**pivot**数值相同的元素，则可以跳过这些元素，以减少排序时间：

```python
#    quick_sort(alist, start, low-1)
#    quick_sort(alist, low+1, end)
    """
    优化排序区间: 跳过与pivot相同的元素
    """
    cur = low - 1
    while cur > start and alist[cur] == alist[low]:
        cur -= 1
    quick_sort(alist, start, cur)
    cur = low + 1
    while cur < end and alist[cur] == alist[low]:
        cur += 1
    quick_sort(alist, cur, end)
```


## （6）归并排序
**归并排序**是采用分治法的一个非常典型的应用。归并排序的思想就是先递归分解数组，再合并数组。

将数组分解最小之后，然后合并两个有序数组，基本思路是比较两个数组的最前面的数，谁小就先取谁，取了后相应的指针就往后移一位。然后再比较，直至一个数组为空，最后把另一个数组的剩余部分复制过来即可。

![](https://pic.imgdb.cn/item/640301f7f144a0100777433f.gif)

```python
def merge_sort(alist):
    if len(alist) <= 1:
        return alist
    num = len(alist)//2
    left = merge_sort(alist[0:num])
    right = merge_sort(alist[num:])
    return merge(left, right)
	
def merge(left, right):
    result = []
    l, r = 0, 0
    while l<len(left) and r<len(right):
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += left[l:]  # 切片超界会返回空列表
    result += right[r:]
    return result
```

时间复杂度：
- 最优时间复杂度：$O(n\log(n))$
- 最坏时间复杂度：$O(n\log(n))$
- 稳定性：稳定

归并排序会产生一个新的数组，牺牲空间复杂度减少时间复杂度。

## （7）堆排序

**堆排序（Heapsort）**是指利用**堆**这种数据结构所设计的一种排序算法。堆是一个近似完全二叉树的结构，其子结点的键值或索引总是小于（或者大于）它的父节点。

堆排序的平均时间复杂度为$Ο(n\log n)$。使用最大堆进行堆排序的步骤：
1. 根据数组创建一个最大堆$H[0,...,n-1]$；
2. 把堆首（最大值）和堆尾互换；
3. 把堆的尺寸缩小**1**：$H'[0,...,n-2]$，并把新的数组顶端数据下沉到相应位置；
4. 重复步骤**2**，直到堆的尺寸为**1**。

![](https://pic1.imgdb.cn/item/6472bb85f024cca173089c66.gif)

```python
def buildMaxHeap(arr):
    for i in range(arrLen//2, -1, -1):
        # 把节点i下沉到对应位置
        heapify(arr, i)

def heapify(arr, i):
    left, right = 2*i+1, 2*i+2
    largest = i
    if left < arrLen and arr[left] > arr[largest]:
        largest = left
    if right < arrLen and arr[right] > arr[largest]:
        largest = right
    
    # 若子节点的值大于父节点i
    if largest != i:
        swap(arr, i, largest)
        heapify(arr, largest)

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

def heapSort(arr):
    # 记录数组长度
    global arrLen
    arrLen = len(arr)
    # 建立最大堆
    buildMaxHeap(arr)
    # 交换堆顶和堆底，并下沉新的堆顶
    for i in range(len(arr)-1, 0, -1):
        swap(arr, 0, i)
        arrLen -= 1
        heapify(arr, 0)
    return arr
```

## （8）计数排序

**计数排序 (Counting Sort)**的核心在于将输入的数据值转化为键存储在额外开辟的数组空间中。作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。

当输入的元素是$n$个$0$到$k$之间的整数时，它的运行时间是$O(n + k)$。计数排序不是比较排序，排序的速度快于任何比较排序算法。由于用来计数的数组$C$的长度取决于待排序数组中数据的范围（等于待排序数组的最大值与最小值的差加上$1$），这使得计数排序对于数据范围很大的数组，需要大量时间和内存。

计数排序算法的步骤如下：
1. 找出待排序的数组中最大和最小的元素
2. 统计数组中每个值为$i$的元素出现的次数，存入数组$C$的第$i$项
3. 对所有的计数累加（从$C$中的第一个元素开始，每一项和前一项相加）
4. 反向填充目标数组：将每个元素$i$放在新数组的第$C(i)$项，每放一个元素就将$C(i)$减去$1$

![](https://pic.imgdb.cn/item/640a9b2bf144a010071e8f56.gif)

```python
def count_sort(alist):
    minValue, maxValue = alist[0], alist[0]
    for elem in alist:
        if elem < minValue:
            minValue = elem
        elif elem > maxValue:
            maxValue = elem
    bucketLen = maxValue-minValue+1
    bucket = [0]*bucketLen
    for elem in alist:
        bucket[elem-minValue] += 1
    cur = 0
    for i in range(bucketLen):
        while bucket[i] > 0:
            alist[cur] = i+minValue
            bucket[i] -= 1
            cur += 1
```

时间复杂度：
- 最优时间复杂度：$O(n+k)$
- 最坏时间复杂度：$O(n+k)$
- 稳定性：稳定

## （9）桶排序

**桶排序**是计数排序的升级版。它利用了函数的映射关系，高效与否的关键就在于这个映射函数的确定。为了使桶排序更加高效，需要做到两点：
1. 在额外空间充足的情况下，尽量增大桶的数量；
2. 使用的映射函数能够将输入的$N$个数据均匀的分配到$K$个桶中。

对于桶中元素的排序，选择何种比较排序算法对于性能的影响至关重要。什么时候最快：当输入的数据可以均匀的分配到每一个桶中；什么时候最慢：当输入的数据被分配到了同一个桶中。

桶排序的时间复杂度和空间复杂度都是$O(N)$，桶排序是稳定的。桶排序的应用场景：通常如果是整数数组还可以用计数排序来计数统计然后排序，但是如果是一个连续空间内的排序，即统计的是一个浮点类型的数组成的数组，那么就无法开辟一个对应的空间使其一一对应的存储。此时，我们需要新建一个带有存储范围的空间，来存储一定范围内的元素。

![](https://pic1.imgdb.cn/item/64734820f024cca173c3b147.jpg)

```python
def bucketSort(arr, max_num):
    buf = {i: [] for i in range(int(max_num)+1)}
    arrLen = len(arr)
    for i in range(arrLen):
        num = arr[i]
        buf[int(num)].append(num)  # 将相应范围内的数据加入到[]中
    arr = []
    for i in range(len(buf)):
        if buf[i]:
            arr.extend(sorted(buf[i]))  # 这里还需要对一个范围内的数据进行排序，然后再进行输出
    return arr
```

## （10）基数排序

**基数排序**是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。

### 基数排序 vs 计数排序 vs 桶排序

这三种排序算法都利用了桶的概念，但对桶的使用方法上有明显差异：
- 基数排序：根据键值的每位数字来分配桶；
- 计数排序：每个桶只存储单一键值；
- 桶排序：每个桶存储一定范围的数值。

![](https://pic1.imgdb.cn/item/64734985f024cca173c52e77.gif)


```python
def radixSort(nums):
    # 遍历数组获取数组最大值和最大值对应的位数
    maxValue = max(nums)
    # 迭代次数
    iterCount = len(str(maxValue))
    for i in range(iterCount):
        # 定义桶，大小为10，对应0-9
        bucket = [[] for _ in range(10)]
        for n in nums:
            index = (n//10**i)%10
            bucket[index].append(n)
        # nums数组清零，并合并桶内元素至nums
        nums.clear()
        for b in bucket:
            nums.extend(b)
    return nums
```



# 4. 搜索
**搜索**是在一个项目集合中找到一个特定项目的算法过程。通常搜索的答案是真的或假的，即表示该项目是否存在。搜索的几种常见方法：顺序查找、二分法查找、二叉树查找、哈希查找。

## （1）二分查找
**二分查找**又称折半查找，优点是比较次数少，查找速度快，平均性能好；其缺点是要求待查表为有序表，且插入删除困难。因此，折半查找方法适用于不经常变动而查找频繁的有序列表。

首先，假设表中元素是按升序排列，将表中间位置记录的关键字与查找关键字比较，如果两者相等，则查找成功；否则利用中间位置记录将表分成前、后两个子表，如果中间位置记录的关键字大于查找关键字，则进一步查找前一子表，否则进一步查找后一子表。重复以上过程，直到找到满足条件的记录，使查找成功，或直到子表不存在为止，此时查找不成功。
![](https://pic.imgdb.cn/item/6403064ef144a010077f2d8d.jpg)

递归实现：
```python
def binary_search(alist, item):
    n = len(alist)
    if n == 0:
        return False
    else:
        mid = n//2
        if alist[mid] == item:
            return True
        elif alist[mid] > item:
            return binary_search(alist[:mid-1], item)
        else:
            return binary_search(alist[mid+1:], item)
```

非递归实现（使用双指针）：
```python
def binary_search(alist, item):
    first = 0
    last = len(alist)-1
    while first <= last:
        mid = (first+last)//2
        if alist[mid] == item:
            return True
        elif alist[mid] > item:
            last = mid - 1
        else:
            first = mid + 1
    return False
```

时间复杂度：
- 最优时间复杂度：$O(1)$
- 最坏时间复杂度：$O(\log(n))$



# 5. 树
**树（tree）**是一种抽象数据类型（**ADT**）或是实作这种抽象数据类型的数据结构，用来模拟具有树状结构性质的数据集合。它是由$n$（$n\geq 1$）个有限节点组成一个具有层次关系的集合。把它叫做“树”是因为它看起来像一棵倒挂的树，也就是说它是根朝上，而叶朝下的。

树具有以下特点：
- 每个节点有零个或多个**子节点**；
- 没有父节点的节点称为**根节点**；
- 每一个非根节点有且只有一个**父节点**；
- 除了根节点外，每个子节点可以分为多个不相交的**子树**。

![](https://pic.downk.cc/item/5e899db8504f4bcb04e5023c.jpg)

### 树的术语
- 节点的**度**：一个节点含有的子树的个数称为该节点的度；
- 树的**度**：一棵树中，最大的节点的度称为树的度；
- **叶节点**或终端节点：度为零的节点；
- 父亲节点或**父节点**：若一个节点含有子节点，则这个节点称为其子节点的父节点；
- 孩子节点或**子节点**：一个节点含有的子树的根节点称为该节点的子节点；
- **兄弟节点**：具有相同父节点的节点互称为兄弟节点；
- 节点的**层次**：从根开始定义起，根为第1层，根的子节点为第2层，以此类推；
- 树的**高度**或**深度**：树中节点的最大层次；
- **堂兄弟节点**：父节点在同一层的节点互为堂兄弟；
- 节点的**祖先**：从根到该节点所经分支上的所有节点；
- **子孙**：以某节点为根的子树中任一节点都称为该节点的子孙；
- **森林**：由$m$（$m\geq 0$）棵互不相交的树的集合称为森林。

### 树的种类
- **无序树**：树中任意节点的子节点之间没有顺序关系，这种树称为无序树，也称为自由树；
- **有序树**：树中任意节点的子节点之间有顺序关系，这种树称为有序树；
- **二叉树**：每个节点最多含有两个子树的树称为二叉树；
- **霍夫曼树**（用于信息编码）：带权路径最短的二叉树称为哈夫曼树或最优二叉树；
- **B树**：一种对读写操作进行优化的自平衡的二叉查找树，能够保持数据有序，拥有多余两个子树。

### 树的存储
- **顺序存储**: 将数据结构存储在固定的数组中，虽然在遍历速度上有一定的优势，但因所占空间比较大，是非主流二叉树。
![](https://pic.downk.cc/item/5e899f3a504f4bcb04e66c0d.jpg)

- **链式存储**: 二叉树通常以链式存储。
![](https://pic.downk.cc/item/5e899f72504f4bcb04e6996b.jpg)

由于对节点的个数无法掌握，常见树的存储表示都转换成二叉树进行处理，子节点个数最多为2。

### 树的应用场景
- 路由协议就是使用了树的算法;
- **mysql**数据库索引;
- 文件系统的目录结构;
- 很多经典的**AI**算法都是树搜索，如机器学习中的**decision tree**;
- **xml**，**html**的解析器，不可避免用到树。
![](https://pic.downk.cc/item/5e899fbf504f4bcb04e6d2db.jpg)

## （1）二叉树
**二叉树**是每个节点最多有两个子树的树结构。通常子树被称作“左子树”（**left subtree**）和“右子树”（**right subtree**）。

### 二叉树的分类：
- **完全二叉树**：对于一颗二叉树，假设其深度为$d$($d>1$)。除了第$d$层外，其它各层的节点数目均已达最大值，且第$d$层所有节点从左向右连续地紧密排列，这样的二叉树被称为完全二叉树；
- **满二叉树**：所有叶节点都在最底层的完全二叉树;
- **平衡二叉树（AVL树）**：当且仅当任何节点的两棵子树的高度差不大于$1$的二叉树；
- **排序二叉树**（**二叉查找树**，**Binary Search Tree**），也称二叉搜索树、有序二叉树。

(1) 完全二叉树:若设二叉树的高度为$h$，除第 $h$ 层外，其它各层 ($1$～$h-1$) 的结点数都达到最大个数，第$h$层有叶子结点，并且叶子结点都是从左到右依次排布，这就是完全二叉树。
![](https://pic.downk.cc/item/5e89a0de504f4bcb04e7abb1.jpg)

(2) 满二叉树:除了叶结点外每一个结点都有左右子叶且叶子结点都处在最底层的二叉树。
![](https://pic.downk.cc/item/5e89a0fb504f4bcb04e7c300.jpg)

### 二叉树的性质：
1. 在二叉树的第$i$层上至多有$2^{i-1}$个结点（$i>0$）;
2. 深度为$k$的二叉树至多有$2^k-1$个结点（$k>0$）;
3. 对于任意一棵二叉树，如果其叶结点数为$N_0$，而度数为$2$的结点总数为$N_2$，则$N_0=N_2+1$;
4. 具有n个结点的完全二叉树的深度必为 $\log_2(n+1)$;
5. 对完全二叉树，若从上至下、从左至右编号，则编号为$i$的结点，其左孩子编号必为$2i$，其右孩子编号必为$2i＋1$；其双亲的编号必为$i/2$（$i＝1$ 时为根,除外）。

### 二叉树的创建：
定义节点类：
```python
class Node(object):
    def __init__(self, elem):
        self.elem = elem
        self.lchild = None
        self.rchild = None
```

创建树的类，并给**root**根节点：
```python
class Tree(object):
    def __init__(self, root = None):
        self.root = root

    def add(self, elem):
    """为树添加节点，采用广度优先遍历"""
        node = Node(elem)
		
        if self.root is None:
            self.root = node
            return
			
        queue = [self.root]  # 维护一个队列
        while queue:  # 注意bool([None]) == True,但bool([]) == False
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return
            elif cur_node.rchild is None:
                cur_node.rchild = node
                return
            else:
                queue.append(cur_node.lchild)
                queue.append(cur_node.rchild)
```

### 二叉树的遍历：
树的遍历是树的一种重要的运算。所谓遍历是指对树中所有结点的信息的访问，即依次对树中每个结点访问一次且仅访问一次，我们把这种对所有节点的访问称为**遍历（traversal）**。

树的两种重要的遍历模式是**深度优先遍历**和**广度优先遍历**,深度优先一般用**递归**，广度优先一般用**队列**。一般情况下能用递归实现的算法大部分也能用堆栈来实现。

**(1) 深度优先遍历**

对于一颗二叉树，**深度优先搜索(Depth First Search)**是沿着树的深度遍历树的节点，尽可能深的搜索树的分支。

深度遍历有重要的三种方法。这三种方式常被用于访问树的节点，它们之间的不同在于访问每个节点的次序不同。这三种遍历分别叫做**先序遍历（preorder）**，**中序遍历（inorder）**和**后序遍历（postorder）**。

- **先序遍历**：根节点->左子树->右子树

在先序遍历中，我们先访问根节点，然后递归使用先序遍历访问左子树，再递归使用先序遍历访问右子树。

使用递归法实现先序遍历：

```python
def preorder(node):
    if node is None:
        return
    print(node.elem)
    preorder(node.lchild)
    preorder(node.rchild)
```

- **中序遍历**：左子树->根节点->右子树

在中序遍历中，我们递归使用中序遍历访问左子树，然后访问根节点，最后再递归使用中序遍历访问右子树。

使用递归法实现中序遍历：

```python
def inorder(node):
    if node is None:
        return
    inorder(node.lchild)
    print(node.elem)
    inorder(node.rchild)
```

- **后序遍历**：左子树->右子树->根节点

在后序遍历中，我们先递归使用后序遍历访问左子树和右子树，最后访问根节点。

使用递归法实现后序遍历：

```python
def postorder(node):
    if node is None:
        return
    postorder(node.lchild)
    postorder(node.rchild)
    print(node.elem)
```

![](https://pic.downk.cc/item/5e89ba55504f4bcb0400149a.jpg)

**思考**：哪两种遍历方式能够唯一的确定一颗树？？？

解答：“**中序+先序**”或“**中序+后序**”能够确定树。

**(2) 广度优先遍历**

从树的**root**开始，从上到下从从左到右遍历整个树的节点。

利用队列实现树的层次遍历:

```python
def breadth_travel(tree):
    if tree is None:
        return
		
    queue = [tree]
    while queue:
        cur_node = queue.pop(0)
        print(cur_node.elem)
        if cur_node.lchild is not None:
            queue.append(cur_node.lchild)
        if cur_node.rchild is not None:
            queue.append(cur_node.rchild)
```

![](https://pic.downk.cc/item/5e89b9f2504f4bcb04ffb5f3.jpg)

## （2）堆

**堆（heap）**是一个完全二叉树，其每个节点的值总是大于等于（或小于等于）子节点的值，分别称为最大堆（大根堆、大顶堆）和最小堆（小根堆、小顶堆）。

![](https://pic.imgdb.cn/item/642ce166a682492fcc944d94.jpg)

在实际实现堆时，通常用一个**数组**而不是用指针建立一个树。这是因为堆是**完全二叉树**，所以用数组表示时，位置 $i$ 节点的父节点位置一定为 $i/2$，而它的两个子节点的位置又一定分别为 $2i$ 和 $2i+1$。注意数组的第一个索引 $0$ 空着不用：

![](https://pic.imgdb.cn/item/642ce23da682492fcc95799c.jpg)

堆的主要应用有两个，首先是一种排序方法「堆排序」，第二是一种很有用的数据结构「优先级队列」。

### ⚪ 最大堆的实现

下面以最大堆为例，给出其代码实现。通过将算法中的大于号和小于号互换，我们也可以得到一个最小堆。

首先定义一个堆类：

```python
class heap(object):
    def __init__(self):
        __heap = [0] # 索引0不用
        __N = 0 # 当前元素个数
    
    # 父节点的索引
    def parent(self, root):
        return root//2
    
    # 左子节点的索引
    def left(self, root):
        return root*2
    
    # 右子节点的索引
    def right(self, root):
        return root*2 + 1

    # 返回最大值
    def max(self):
        return __heap[1]
```

堆主要有两种方法，分别是`insert`插入一个元素和`delMax`删除最大元素（如果底层用最小堆，那么就是`delMin`）。最大堆的每个节点都比它的两个子节点大，但是在插入元素和删除元素时，难免破坏堆的性质，这时需要通过**上浮**和**下沉**这两个操作来恢复堆的性质。

对于最大堆，会破坏堆性质的有两种情况：
1. 如果某个节点比它的子节点（中的一个）小，那么该节点不应该做父节点，需要交换这两个节点；交换后还可能比它新的子节点（中的一个）小，因此需要不断地进行比较和交换操作。这就是对该节点的**下沉(sink)**。
2. 如果某个节点比它的父节点大，那么该节点不应该做子节点，需要交换这两个节点；交换后还可能比它新的父节点大，因此需要不断地进行比较和交换操作。这就是对该节点的**上浮(swim)**。

上浮的代码实现：

```python
    # 上浮
    def swim(self, k):
        # 如果浮到堆顶，就不能再上浮了
        while k > 1 and self.__heap[self.parent(k)] < self.__heap[k]:
            # 如果第 k 个元素比上层大，将 k 换上去
            self.__heap[self.parent(k)], self.__heap[k] = \
                self.__heap[k], self.__heap[self.parent(k)]
            k = self.parent(k)
```

下沉的代码实现：

```python
    # 下沉
    def sink(self, k):
        # 如果沉到堆底，就沉不下去了
        while self.left(k) <= self.__N:
            # 先假设左边节点较大
            older = self.left(k)
            # 如果右边节点存在，比一下大小
            if self.right(k) <= self.__N and self.__heap[self.right(k)] > self.__heap[older]:
                older = self.right(k)
            # 结点 k 比俩孩子都大，就不必下沉了
            if self.__heap[k] > self.__heap[older]:
                break
            # 否则不符合最大堆的结构，下沉 k 结点
            self.__heap[k], self.__heap[older] = self.__heap[older], self.__heap[k]
            k = older
```

`insert`方法先把要插入的元素添加到堆底的最后，然后让其上浮到正确位置。

```python
    # 插入
    def insert(self, elem):
        self.__N += 1
        # 先把新元素加到最后
        self.__heap.append(elem)
        # 然后让它上浮到正确的位置
        self.swim(self.__N)
```

`delMax`方法先把堆顶元素 **A** 和堆底最后的元素 **B** 对调，然后弹出 **A**，最后让 **B** 下沉到正确位置。

```python
    # 删除最大值
    def delMax(self):
        # 最大堆的堆顶就是最大元素
        # 把这个最大元素换到最后，删除之
        self.__heap[1], self.__heap[self.__N] = self.__heap[self.__N], self.__heap[1]
        __max = self.__heap.pop()
        self.__N -= 1
        # 让 heap[1] 下沉到正确位置
        self.sink(1)
        return __max
```

最大堆插入和删除元素的时间复杂度为 $O(\log K)$，$K$ 为当前二叉堆中的元素总数。时间复杂度主要花费在 **sink** 或者 **swim** 上，而不管上浮还是下沉，最多也就是树（堆）的高度，也就是 $\log$ 级别。

## （3）四叉树与八叉树

### ⚪ 四叉树

**四叉树（Quadtree）**或四元树也被称为**Q**树（**Q-Tree**），是二叉树在二维的引申，广泛应用于图像处理、空间数据索引、**2D**中的快速碰撞检测、存储稀疏数据等。

四叉树的定义是每个节点下至多可以有四个子节点。通常把一部分二维空间细分为四个象限或区域并，把该区域里的相关信息存入到四叉树节点中。这个区域可以是正方形、矩形或是任意形状。

![](https://pic.imgdb.cn/item/649d18bf1ddac507cc069b58.jpg)

四叉树的每一个节点代表一个矩形区域（如上图黑色的根节点代表最外围黑色边框的矩形区域），每一个矩形区域又可划分为四个小矩形区域，这四个小矩形区域作为四个子节点所代表的矩形区域。

### ⚪ 八叉树

**八叉树（Octree）**是二叉树在三维的引申，主要应用于**3D**图形处理、游戏编程、激光雷达点云处理等。

八叉树中每个内部节点最多可以有**8**个子节点。就像二叉树把空间分成两个部分一样，八叉树把空间最多分成**8**个部分，用于存储空间大的三维点。如果八叉树的所有内部节点恰好包含**8**个子节点，则称为全八叉树。

![](https://pic.imgdb.cn/item/649d1a2a1ddac507cc0898f7.jpg)

八叉树的构建流程：
1. 通过设置分辨率来计算最大递归深度，分辨率描述最低一级的八叉树的最小体素的尺寸，因此八叉树的深度是分辨率和点云数据的函数。
2. 找出场景的最大尺寸，并以此尺寸建立第一个立方体。
3. 依序将单位元元素丢入能被包含且没有子节点的立方体。
4. 若没达到最大递归深度，就进行细分八等份，再将该立方体所装的单位元元素全部分担给八个子立方体。
5. 若发现子立方体所分配到的单位元元素数量不为零且跟父立方体是一样的，则该子立方体停止细分，因为跟据空间分割理论，细分的空间所得到的分配必定较少，若是一样数目，则再怎么切数目还是一样，会造成无穷切割的情形。
6. 重复3，直到达到最大递归深度。
