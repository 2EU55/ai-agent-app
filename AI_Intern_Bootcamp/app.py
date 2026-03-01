import json
from multiprocessing import Value
import os
import uuid
import streamlit as st #导入模块

DATA_FILE = 'todos.json'#设置文件名

def load_todos():#读取函数
    if not os.path.exists(DATA_FILE):#如果不存在
        return []#返回空列表
    try:
        with open(DATA_FILE,'r',encoding='utf-8') as f:#如果存在
            data =  json.load(f)#以json格式读取
    except Exception:
        return []
    return data if isinstance(data,list) else []#返回数据格式如果为list则返回list 如果不是则返回空列表

def save_todos(todos):#保存函数
    with open(DATA_FILE,'w',encoding='utf-8') as f:
        json.dump(todos,f,ensure_ascii=False,indent=2)#以json格式保存ascii不转义缩进为2

st.set_page_config(page_title='Todo', layout='centered')

if 'todos' not in st.session_state:#检查初始状态
    st.session_state.todos = load_todos()#第一次打开为空则读取本地放入网页内存，点击按钮会重跑，避免每次重跑都重新读取load覆盖内存数据

if 'editing_id' not in st.session_state:
    st.session_state.editing_id = None

st.title('Todo')#标题
st.caption(f'数据文件：{os.path.abspath(DATA_FILE)}')

title = st.text_input('新增代办')#输入框
if st.button('添加'):#触发按钮
    if title.strip():#如果输入框输入不为空，顺便去掉前后空格
        st.session_state.todos.append(
            {'id':str(uuid.uuid4()),'title':title.strip(),'done':False}
        )#加入列表设置唯一id标题为输入，按钮状态为未选中
        save_todos(st.session_state.todos)#保存列表
        st.rerun()

st.divider()#分割线

for i,todo in enumerate(st.session_state.todos):#把列表数据变成一行行的UI
    col1,col2,col3 = st.columns([1,0.25,0.25])#定义显示区域
    with col1:#区域1
        if st.session_state.editing_id == todo["id"]:
            new_title = st.text_input(
                "编辑标题",
                value=todo["title"],
                key=f'edit_title_{todo["id"]}'
            )

            c1,c2 = st.columns([0.3,0.3])

            with c1:
                if st.button('保存',key=f'save_{todo["id"]}'):
                    if new_title.strip():
                        st.session_state.todos[i]["title"] = new_title.strip()
                        save_todos(st.session_state.todos)
                        st.session_state.editing_id = None
                        st.rerun()
            
            with c2:
                if st.button("取消",key=f'cancel_{todo["id"]}'):
                    st.session_state.editing_id = None
                    st.rerun()
        else:
            done = st.checkbox(todo['title'],value=bool(todo['done']),key=f'done_{todo["id"]}')#复选按钮
            if done != todo['done']:#判断done状态有没有改变
                st.session_state.todos[i]['done'] = done#更新状态
                save_todos(st.session_state.todos)#保存
                st.rerun()
    with col2:#区域2
        if st.button('编辑',key=f'edit_{todo["id"]}'):
            st.session_state.editing_id = todo["id"]
            st.rerun()
    with col3:
        if st.button('删除',key=f'del_{todo["id"]}'):#删除按钮
            st.session_state.todos.pop(i)#删除
            save_todos(st.session_state.todos)#保存
            st.rerun()

