# -*- coding: utf-8 -*-

import sys
import time
import os
from functools import wraps
from typing import (List)
from typing import Union

import uiautomation as auto
sys.path.append(os.path.dirname(__file__))
import UiGetControl

# TextControl = Union[auto.ComboBoxControl, auto.DataItemControl, auto.DocumentControl, auto.EditControl,
#                     auto.HyperlinkControl, auto.ListItemControl, auto.ProgressBarControl, auto.SliderControl, auto.SpinnerControl]


def _try_msg(f):
    """作为参数的函数执行过程中
    1）若执行出错，则打印并返回 (None,err_msg）；
    2）若正常，则返回(f执行结果,‘’）
    :param f:
    :return f,err_msg:
    """
    @wraps(f)
    def wrapped_f(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            if type(result) is tuple:
                return result
            else:
                return result, ''
        except Exception as err:
            info = sys.exc_info()[2].tb_frame.f_back
            temp = "filename:{}\nlines:{}\tfunction:{}\terror:{}"
            err_msg = temp.format(info.f_code.co_filename, info.f_lineno,
                                  f.__name__, repr(err))
            print(err_msg)
            return None, err_msg

    return wrapped_f


@_try_msg
def get_children(parent: auto.Control = None) -> List[auto.Control]:
    """获取子元素列表

    :param parent:
    :return: 子元素列表
    """
    if parent is None:
        parent = auto.GetRootControl()
    return parent.GetChildren()


@_try_msg
def get_first_child(parent: auto.Control = None) -> auto.Control:
    """获取第一个子元素

    :param parent:
    :return: 第一个子元素
    """
    if parent is None:
        parent = auto.GetRootControl()
    return parent.GetFirstChildControl()


@_try_msg
def get_last_child(parent: auto.Control = None) -> auto.Control:
    """获取倒一个子元素

    :param parent:
    :return: 倒一个子元素
    """
    if parent is None:
        parent = auto.GetRootControl()
    return parent.GetLastChildControl()


@_try_msg
def get_parent_control(control: auto.Control) -> auto.Control:
    """获取父元素

    :param parent:
    :return: 父元素
    """
    return control.GetParentControl()


@_try_msg
def exist_control(control: auto.Control) -> bool:
    """元素存在与否判断

    :param parent:
    :return: bool值
    """
    return control.Exists()


@_try_msg
def set_text(control: auto.Control, text: str) -> bool:
    """设置文本

    :param control:
    :param text:
    :return:
    """
    return control.GetPattern(auto.PatternId.ValuePattern).SetValue(text)


@_try_msg
def get_text(control: auto.Control) -> str:
    """获取文本

    :param control:
    :return:
    """
    return control.GetPattern(auto.PatternId.ValuePattern).Value


@_try_msg
def click(control: auto.Control, x: int = None, y: int = None) -> None:
    """鼠标单击左键

    :param control:
    :param x:
    :param y:
    :return:
    """
    switch_to_this_Window(control)
    return control.Click(x, y)


@_try_msg
def double_click(control: auto.Control, x: int = None, y: int = None) -> None:
    """鼠标双击左键

    :param control:
    :param x:
    :param y:
    :return:
    """
    switch_to_this_Window(control)
    return control.DoubleClick(x, y)


@_try_msg
def right_click(control: auto.Control, x: int = None, y: int = None) -> None:
    """鼠标单击右键

    :param control:
    :param x:
    :param y:
    :return:
    """
    switch_to_this_Window(control)
    return control.RightClick(x, y)


@_try_msg
def middle_click(control: auto.Control, x: int = None, y: int = None) -> None:
    """鼠标单击中键

    :param control:
    :param x:
    :param y:
    :return:
    """
    switch_to_this_Window(control)
    return control.MiddleClick(x, y)


@_try_msg
def switch_to_this_Window(control: auto.Control,
                          Mmize='',
                          set_top_most=None) -> None:
    """切换窗口

    :param control: 窗口对象
    :param Mmize: 输入'Maximize'表示最大化，输入'Minimize'表示最小化，其他值时不做控制，默认不做控制
    :param set_top_most:是否设置为最上层窗口，默认为None，表示不设置。输入bool值 True 或者FALSE， 若设置为最上层窗口，则不会被其他窗口遮挡
    :return:
    """
    result = None
    handle = control.NativeWindowHandle
    # print(handle)
    if not handle:
        while not handle:
            control = control.GetParentControl()
            handle = control.NativeWindowHandle
    # print(handle)
    if handle:
        if set_top_most is not None:
            auto.SetWindowTopmost(handle, set_top_most)
        time.sleep(auto.OPERATION_WAIT_TIME)
        auto.SwitchToThisWindow(handle)
        time.sleep(auto.OPERATION_WAIT_TIME)
        if Mmize == 'Maximize':
            auto.ShowWindow(handle, auto.SW.Maximize)
            time.sleep(auto.OPERATION_WAIT_TIME)
        if Mmize == 'Minimize':
            auto.ShowWindow(handle, auto.SW.Minimize)
    return result


@_try_msg
def set_focus(control: auto.Control) -> bool:
    """获取焦点

    :return:bool值 True 或 False
    """
    return control.SetFocus()


@_try_msg
def get_control_rectangle(control: auto.Control) -> tuple:
    """获取control位置和大小

    :return:left, top, width, height 
    """
    rect = control.BoundingRectangle
    left = rect.left
    top = rect.top
    right = rect.right
    bottom = rect.bottom
    width = right - left
    height = bottom - top
    return [left, top, width, height]


@_try_msg
def get_next_sibling_control(control: auto.Control,
                             bias: int = 1) -> auto.Control:
    """获取第N个下元素

    :param control:control元素
    :param bias:第几个（下元素），默认为1
    :return:下元素，不存在则返回None
    """
    if bias == 0:
        return control
    elif bias < 0:
        return None
    while control:
        control = control.GetNextSiblingControl()
        bias -= 1
        if bias == 0:
            break
    return control


@_try_msg
def get_pre_sibling_control(control: auto.Control,
                            bias: int = 1) -> auto.Control:
    """获取第N个上元素

    :param control:control元素
    :param bias:第几个（下元素），默认为1
    :return:上元素，不存在则返回None
    """
    if bias == 0:
        return control
    elif bias < 0:
        return None
    while control:
        control = control.GetPreviousSiblingControl()
        bias -= 1
        if bias == 0:
            break
    return control


@_try_msg
def set_text_by_pattern(controlPara: str = '',
                        text: str = '',
                        timeout: float = 1,
                        searchTimes: int = 1,
                        parent: auto.Control = None) -> bool:
    """查找并设置文本

    :param controlPara:特征字符串
    :param text:设置的文本值
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return: bool值 True/False
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    return control.GetPattern(auto.PatternId.ValuePattern).SetValue(text)


@_try_msg
def get_text_by_pattern(controlPara: str = '',
                        timeout: float = 1,
                        searchTimes: int = 1,
                        parent: auto.Control = None) -> str:
    """查找并获取文本

    :param controlPara:特征字符串
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return: 文本
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    return control.GetPattern(auto.PatternId.ValuePattern).Value


@_try_msg
def click_by_pattern(controlPara: str = '',
                     x: int = None,
                     y: int = None,
                     timeout: float = 1,
                     searchTimes: int = 1,
                     parent: auto.Control = None) -> None:
    """查找并单击

    :param controlPara:特征字符串
    :param x:横向偏移，默认为None，表示中间位置，若大于0则表示相对于左侧偏移，若小于则表示相对于右侧偏移
    :param y:纵向偏移，默认为None，表示中间位置，若大于0则表示相对于顶部偏移，若小于则表示相对于底部偏移
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return:None
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    switch_to_this_Window(control)
    return control.Click(x, y)


@_try_msg
def double_click_by_pattern(controlPara: str = '',
                            x: int = None,
                            y: int = None,
                            timeout: float = 1,
                            searchTimes: int = 1,
                            parent: auto.Control = None) -> None:
    """查找并双击

    :param controlPara:特征字符串
    :param x:横向偏移，默认为None，表示中间位置，若大于0则表示相对于左侧偏移，若小于则表示相对于右侧偏移
    :param y:纵向偏移，默认为None，表示中间位置，若大于0则表示相对于顶部偏移，若小于则表示相对于底部偏移
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return:None
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    switch_to_this_Window(control)
    return control.DoubleClick(x, y)


@_try_msg
def right_click_by_pattern(controlPara: str = '',
                           x: int = None,
                           y: int = None,
                           timeout: float = 1,
                           searchTimes: int = 1,
                           parent: auto.Control = None) -> None:
    """查找并单击右键

    :param controlPara:特征字符串
    :param x:横向偏移，默认为None，表示中间位置，若大于0则表示相对于左侧偏移，若小于则表示相对于右侧偏移
    :param y:纵向偏移，默认为None，表示中间位置，若大于0则表示相对于顶部偏移，若小于则表示相对于底部偏移
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return:None
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    switch_to_this_Window(control)
    return control.RightClick(x, y)


@_try_msg
def middle_click_by_pattern(controlPara: str = '',
                            x: int = None,
                            y: int = None,
                            timeout: float = 1,
                            searchTimes: int = 1,
                            parent: auto.Control = None) -> None:
    """查找并单击中键

    :param controlPara:特征字符串
    :param x:横向偏移，默认为None，表示中间位置，若大于0则表示相对于左侧偏移，若小于则表示相对于右侧偏移
    :param y:纵向偏移，默认为None，表示中间位置，若大于0则表示相对于顶部偏移，若小于则表示相对于底部偏移
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return:None
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    switch_to_this_Window(control)
    return control.MiddleClick(x, y)


@_try_msg
def switch_to_this_Window_by_pattern(
        controlPara: str = '',
        Mmize='',
        set_top_most=None,
        timeout: float = 1,
        searchTimes: int = 1,
        parent: auto.Control = None) -> auto.Control:
    """查找并切换窗口

    :param controlPara:特征字符串
    :param Mmize: 输入'Maximize'表示最大化，输入'Minimize'表示最小化，默认最大化
    :param set_top_most:是否设置为最上层窗口，默认为None，表示不设置。输入bool值 True 或者FALSE， 若设置为最上层窗口，则不会被其他窗口遮挡
    :param timeout:单次查找超时时间（秒），默认为1s
    :param searchTimes:重试次数，默认为1
    :param parent:父节点，默认为None
    :return:窗口
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    handle = control.NativeWindowHandle
    # print(handle)
    if not handle:
        while not handle:
            control = control.GetParentControl()
            handle = control.NativeWindowHandle
    # print(handle)
    if handle:
        if set_top_most is not None:
            auto.SetWindowTopmost(handle, set_top_most)
        time.sleep(auto.OPERATION_WAIT_TIME)
        auto.SwitchToThisWindow(handle)
        time.sleep(auto.OPERATION_WAIT_TIME)
        if Mmize == 'Maximize':
            auto.ShowWindow(handle, auto.SW.Maximize)
            time.sleep(auto.OPERATION_WAIT_TIME)
        if Mmize == 'Minimize':
            auto.ShowWindow(handle, auto.SW.Minimize)
    return control


@_try_msg
def send_key_by_pattern(
    controlPara,
    key: int,
    timeout: float = 1,
    searchTimes: int = 1,
    parent: auto.Control = None,
) -> None:
    """在control内按键
    先聚焦，然后再按键

    :param control:
    :param key: int, a key code value in class Keys.
    :return:
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    return control.SendKey(key)


@_try_msg
def send_keys_by_pattern(controlPara,
                         text: str,
                         timeout: float = 1,
                         searchTimes: int = 1,
                         parent: auto.Control = None,
                         interval: float = 0.01,
                         charMode: bool = True) -> None:
    """在control内按键
        先聚焦，然后再按键
    Make control have focus first and type keys.
    `self.SetFocus` may not work for some controls, you may need to click it to make it have focus.
    text: str, keys to type, see the docstring of `SendKeys`.
    interval: float, seconds between keys.
    charMode: bool, if False, the text typied is depend on the input method if a input method is on.
    """
    control = UiGetControl.findControl(controlPara, timeout, searchTimes,
                                       parent)
    if control is None:
        return None, '查找不到该control'
    return control.SendKeys(text, interval, charMode)


@_try_msg
def send_key(control: auto.Control, key: int) -> None:
    """在control内按键
    先聚焦，然后再按键

    :param control:
    :param key: int, a key code value in class Keys.
    :return:
    """
    return control.SendKey(key)


@_try_msg
def send_keys(control: auto.Control,
              text: str,
              interval: float = 0.01,
              charMode: bool = True) -> None:
    """在control内按键
        先聚焦，然后再按键
    Make control have focus first and type keys.
    `self.SetFocus` may not work for some controls, you may need to click it to make it have focus.
    text: str, keys to type, see the docstring of `SendKeys`.
    interval: float, seconds between keys.
    charMode: bool, if False, the text typied is depend on the input method if a input method is on.
    text:
    特殊键：{Ctrl}, {Delete} ... 
    组合键：{Ctrl}a  表示 Ctrl+a, 
            {Shift}(123)) 表示 Shift+1+2+3,
    若要按'{'需要转译成'{{'
    """
    return control.SendKeys(text, interval, charMode)


#add by xuly 201130 start
def SetChildrenSelection(comboBox: auto.Control, text: str):
    '''选中下拉菜单
    comboBox 下拉菜单控件
    text 欲选中的菜单名称
    返回：错误信息，成功选中时为空'''
    if comboBox is None:
        return '下拉菜单控件为空'
    comboBox.SetFocus()
    if comboBox.GetPattern(auto.PatternId.ValuePattern) is None:
        return '无法获取下拉菜单信息'
    nowValue = comboBox.GetPattern(auto.PatternId.ValuePattern).Value
    comboBox.Click()
    listItem = comboBox.ListItemControl(searchDepth=2, Name=text)
    if listItem.Exists(maxSearchSeconds=0.5):
        listItem.Click()
    else:
        listItem = auto.ListItemControl(searchDepth=2, Name=text)
        if listItem.Exists(maxSearchSeconds=0.5):
            if nowValue is not None and not listItem.GetParentControl(
            ).ListItemControl(searchDepth=1,
                              Name=nowValue).Exists(maxSearchSeconds=0.1):
                #当直接从UI根元素查找时，通过校验旧值是否在选项的同级可确保没有选错父元素
                return '无法找到菜单项'
            listItem.Click()
        else:
            return '无法找到菜单项'
    return ''


def SetChildrenSelectionByPattern(text: str,
                                  controlPara: str,
                                  timeout: float = 1,
                                  searchTimes: int = 1,
                                  parent: auto.Control = None):
    '''查找并选中下拉菜单
    text 欲选中的菜单名称
    controlPara 下拉菜单标识字符串
    timeout 单次搜索时间(至少1秒)
    searchTimes 搜索次数
    parent 父控件（默认从桌面开始搜索）
    返回：错误信息，成功选中时为空'''
    comboBox = UiGetControl.findControl(controlPara=controlPara,
                                        timeout=timeout,
                                        searchTimes=searchTimes,
                                        parent=parent)
    return SetChildrenSelection(comboBox, text)


def GetComboBoxItems(comboBox: auto.Control):
    '''获取下拉菜单的所有选项
    comboBox 下拉菜单控件
    返回： 选项名称列表，获取失败时返回None'''
    if comboBox is None:
        return None
    itemList = []
    comboBox.SetFocus()
    if comboBox.GetPattern(auto.PatternId.ValuePattern) is None:
        return None
    nowValue = comboBox.GetPattern(auto.PatternId.ValuePattern).Value
    comboBox.Click()
    if nowValue is None:
        return None
    listItem = comboBox.ListItemControl(searchDepth=2, Name=nowValue)
    if not listItem.Exists(maxSearchSeconds=1):
        listItem = auto.ListItemControl(searchDepth=2, Name=nowValue)
    if not listItem.Exists(maxSearchSeconds=1):
        return None
    children = listItem.GetParentControl().GetChildren()
    for child in children:
        itemList.append(child.Name)
    comboBox.SendKeys('{ESC}')
    return itemList


def GetComboBoxItemsByPattern(controlPara: str,
                              timeout: float = 1,
                              searchTimes: int = 1,
                              parent: auto.Control = None):
    '''查找并获取下拉菜单的所有选项
    controlPara 下拉菜单标识字符串
    timeout 单次搜索时间(至少1秒)
    searchTimes 搜索次数
    parent 父控件（默认从桌面开始搜索）
    返回： 选项名称列表，获取失败时返回None'''
    comboBox = UiGetControl.findControl(controlPara=controlPara,
                                        timeout=timeout,
                                        searchTimes=searchTimes,
                                        parent=parent)
    return GetComboBoxItems(comboBox)


#add by xuly 201130 end
#add by xuly 210326 start
@_try_msg
def Invoke(control: auto.Control):
    '''触发控件事件
    control 欲触发事件的控件
    返回：触发成功与否(布尔值)'''
    if control is None:
        return False
    return control.GetPattern(auto.PatternId.InvokePattern).Invoke()

@_try_msg
def FindAndInvoke(controlPara: str,
                   timeout: float = 1,
                   searchTimes: int = 1,
                   parent: auto.Control = None):
    '''查找并触发控件事件
        controlPara 下拉菜单标识字符串
    timeout 单次搜索时间(至少1秒)
    searchTimes 搜索次数
    parent 父控件（默认从桌面开始搜索）
    返回：触发成功与否(布尔值)'''
    control = UiGetControl.findControl(controlPara=controlPara,
                                       timeout=timeout,
                                       searchTimes=searchTimes,
                                       parent=parent)
    if control is None:
        raise Exception('无法找到控件')
    return Invoke(control)

#add by xuly 210326 end


def _test():

    result, err_msg = switch_to_this_Window_by_pattern(
        '{"ClassName":"Notepad","ControlTypeName":"WindowControl","subName":"无标题 - 记事本","searchDepth":1}=>{"ControlTypeName":"TitleBarControl","searchDepth":1}=>{"ControlTypeName":"ButtonControl","Name":"关闭","searchDepth":1}'
    )
    print(err_msg)
    # print(auto.GetRootControl())
    # subprocess.Popen('notepad.exe')
    # 首先从桌面的第一层子控件中找到记事本程序的窗口WindowControl，再从这个窗口查找子控件
    # notepadWindow = auto.WindowControl(searchDepth=1, ClassName='Notepad')
    # a = notepadWindow.BoundingRectangle
    # print(a)
    # notepadWindow = UiGetControl.findControl('{"ClassName":"Notepad","ControlTypeName":"WindowControl","subName":"无标题 - 记事本","searchDepth":1}',5,10)
    # print(notepadWindow)
    # result,err_msg = get_children(notepadWindow)
    # print(result)
    # result,err_msg = click_by_pattern(notepadWindow)
    # print(err_msg)
    # print(result)
    # result,err_msg = get_first_child(result)
    # print(result)
    # print(1111111111)
    # result,err_msg = get_parent_control(result)
    # print(result)
    # result,err_msg = get_parent_control(result)
    # print(result)
    #
    # print(notepadWindow.Name)
    # notepadWindow.SetTopmost(True)
    # notepadWindow.SwitchToThisWindow()
    # notepadWindow.Maximize()
    # sleep(3)
    # notepadWindow.Minimize()
    # notepadWindow.MoveToCenter()
    # 查找notepadWindow所有子孙控件中的第一个EditControl，因为EditControl是第一个子控件，可以不指定深度
    # edit = notepadWindow.EditControl()
    # switch_to_this_Window(notepadWindow,Mmize='Minimize', set_top_most=False)
    # set_focus(edit)
    # notepadWindow.SetFocus()
    # 获取EditControl支持的ValuePattern，并用Pattern设置控件文本为"Hello"
    # set_text(edit, 'jgbasjgb')  # or edit.GetPattern(auto.PatternId.ValuePattern)
    # edit.SendKeys('{Ctrl}{End}{Enter}World')  # 在文本末尾打字
    # 先从notepadWindow的第一层子控件中查找TitleBarControl,
    # 然后从TitleBarControl的子孙控件中找第二个ButtonControl, 即最大化按钮，并点击按钮
    # max_button = notepadWindow.TitleBarControl(Depth=1).ButtonControl(foundIndex=2)
    # click(max_button)
    # sleep(2)
    # double_click(max_button)
    # sleep(2)
    # middle_click(max_button)
    # sleep(2)
    # double_click(max_button)
    # sleep(2)

    # 从notepadWindow前两层子孙控件中查找Name为'关闭'的按钮并点击按钮
    # notepadWindow.ButtonControl(searchDepth=2, Name='关闭').Click()
    # 这时记事本弹出是否保存提示，按热键Alt+N不保存退出。
    # auto.SendKeys('{Alt}n')


# if __name__ == '__main__':
#     _test()
