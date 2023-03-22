###### tags: `data_structure` `computre_sience`
{%hackmd BJrTq20hE %}
# Linked List: 新增資料、刪除資料、反轉  
![](https://i.imgur.com/wtEMNN7.png)  
***
## 定義list node, link list  
```C++=
#include <iostream>
using std::cout;
using std::endl;

class LinkedList;    // 為了將class LinkedList設成class ListNode的friend,
                     // 需要先宣告
class ListNode{
private:
    int data;
    ListNode *next;
public:
    ListNode():data(0),next(0){};
    ListNode(int a):data(a),next(0){};

    friend class LinkedList;
};

class LinkedList{
private:
    // int size;                // size是用來記錄Linked list的長度, 非必要
    ListNode *first;            // list的第一個node
public:
    LinkedList():first(0){};
    void PrintList();           // 印出list的所有資料
    void Push_front(int x);     // 在list的開頭新增node
    void Push_back(int x);      // 在list的尾巴新增node
    void Delete(int x);         // 刪除list中的 int x
    void Clear();               // 把整串list刪除
    void Reverse();             // 將list反轉: 7->3->14 => 14->3->7
};
``` 
freind class: C++提供一種方法，使類別A可以存取類別B放在private區域的類別成員。  
***
## 函式：PrintList()  
第一個要介紹的是PrintList()，功能就是把Linked list中的所有資料依序印出。要印出所有的資料，就必須「逐一訪問(Visiting)」Linked list中的每一個node，這樣的操作又稱為Traversal(尋訪)。  
![](https://i.imgur.com/eU4vISA.png =300x300)  
:::success
建立ListNode *current來表示「目前走到哪一個node」。  
若要對Linked list存取資料，必定是從第一個node開始，所以把current指向first所代表的記憶體位置，current=first。  
--目前first即為node(7)。  
--同時，還能夠知道「下一個node」是指向node(3)。  
在印出current->data，也就是7後，便把current移動到「下一個node」。  
--透過current=current->next，即可把current指向node(3)所在的記憶體位置。  
重複上述步驟，直到current指向Linked list的終點NULL為止，便能印出所有資料。  
:::
```cpp=
void LinkedList::PrintList(){

    if (first == 0) {                      // 如果first node指向NULL, 表示list沒有資料
        cout << "List is empty.\n";
        return;
    }

    ListNode *current = first;             // 用pointer *current在list中移動
    while (current != 0) {                 // Traversal
        cout << current->data << " ";
        current = current->next;
    }
    cout << endl;
}
```
***
## 函式：Push_front  
Push_front()的功能是在Linked list的開頭新增資料。  
:::success
1. 先建立一個新的節點ListNode *newNode，帶有欲新增的資料(23)，如圖二(a)。  
2. 將newNode中的pointer：ListNode *next，指向Linked list的第一個nodefirst，如圖二(b)。  
3. 接著，把first更新成newNode。  
:::
![](https://i.imgur.com/0PMXWKk.png =300x300)  
```cpp=
void LinkedList::Push_front(int x){

    ListNode *newNode = new ListNode(x);   // 配置新的記憶體
    newNode->next = first;                 // 先把first接在newNode後面
    first = newNode;                       // 再把first指向newNode所指向的記憶體位置
}
```
***
## 函式：Push_back  
Push_back()的功能是在Linked list的尾巴新增資料。
:::success
* 先建立一個新的節點ListNode *newNode，帶有欲新增的資料(23)。  
* 先利用如同PrintList()中提過的Traversal，把新建立ListNode*current移動到Linked list的尾端，node(14)，如圖三(a)。  
有些資料結構會在class LinkedList中新增一項ListNode *last，記錄Linked list的最後一個node，那麼，Push_back()就不需要Traversal，可以在O(1)時間內完成。  
若沒有ListNode *last，就需要O(N)的Traversal。  
* 接著把current的next pointer指向newNode，如圖三(b)。  
:::
![](https://i.imgur.com/5bxHDRx.png =350x300)
```cpp=
void LinkedList::Push_back(int x){

    ListNode *newNode = new ListNode(x);   // 配置新的記憶體

    if (first == 0) {                      // 若list沒有node, 令newNode為first
        first = newNode;
        return;
    }

    ListNode *current = first;
    while (current->next != 0) {           // Traversal
        current = current->next;
    }
    current->next = newNode;               // 將newNode接在list的尾巴
}
```
***
參考資料:  
(1)  
http://alrightchiu.github.io/SecondRound/linked-list-xin-zeng-zi-liao-shan-chu-zi-liao-fan-zhuan.html  
(2)  
https://crmne0707.pixnet.net/blog/post/317520732-c%2B%2B-%E9%A1%9E%E5%88%A5-class-friend%E7%9A%84%E7%94%A8%E6%B3%95  
(3)  

