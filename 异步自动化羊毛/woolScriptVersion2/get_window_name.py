import time
import uiautomation as auto


# 获取每一个商户所在的根元素
def get_root_control():
    documentControl = auto.DocumentControl(ClassName='Chrome_RenderWidgetHostHWND')
    # 层层搜索找到目标元素
    documentControl_f_son_control = documentControl.GetChildren()[0]
    f_son_control = documentControl_f_son_control.GetChildren()

    target_list_control = None
    for each_control in f_son_control:
        son_control = each_control.GetChildren()
        if len(son_control) == 1:
            if son_control[0].ControlTypeName == 'ListControl':
                target_list_control = son_control[0]
                break

    return target_list_control


def get_comment_pic():
    mainWindow = auto.PaneControl(ClassName='Chrome_WidgetWin_1')
    print('mainWindow Name:', mainWindow.Name)

    if mainWindow.Exists(3, 1):
        handle = mainWindow.NativeWindowHandle
        auto.SwitchToThisWindow(handle)

    target_list_control = get_root_control()
    business_num = len(target_list_control.GetChildren()) - 1
    print('business_num:', business_num)

    # 页数
    PageNum = 8
    # 获取xxxx评论元素的大小并截图
    for i in range(business_num):

        # 第一层，第二层，第三层数据
        layer_1_ele = target_list_control.GetChildren()[i]
        layer_2_ele = layer_1_ele.GetChildren()[1]
        layer_2_son_ele = layer_2_ele.GetChildren()

        layer_3_ele = None
        for each in layer_2_son_ele:
            if "条评价" in each.Name:
                layer_3_ele = each
        if layer_3_ele is None:
            raise Exception('获取元素失败')
        print(layer_3_ele)

        layer_3_ele_bottom = layer_3_ele.BoundingRectangle.bottom
        print('layer_3_ele_bottom:', layer_3_ele_bottom)

        if layer_3_ele_bottom > 0:
            time.sleep(1)
            layer_3_ele.CaptureToImage('./pic/%d%d.png' % (PageNum, i))

        if layer_3_ele_bottom <= 0:
            print('目标元素未显示')
            # auto.WheelDown()
            # auto.mouse_event(auto.MouseEventFlag.Wheel, 0, 0, -435, 0)
            auto.SendKeys("{PAGEDOWN}")
            time.sleep(2)
            target_list_control = get_root_control()
            layer_1_ele = target_list_control.GetChildren()[i]
            layer_2_ele = layer_1_ele.GetChildren()[1]
            layer_2_son_ele = layer_2_ele.GetChildren()

            layer_3_ele = None
            for each in layer_2_son_ele:
                if "条评价" in each.Name:
                    layer_3_ele = each
            if layer_3_ele is None:
                raise Exception('获取元素失败')
            print(layer_3_ele)

            layer_3_ele.CaptureToImage('./pic/%d%d.png' % (PageNum, i))


get_comment_pic()
