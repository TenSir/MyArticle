import os
import psutil
import xlwings as xw



def kill_excel_by_pid():
    # 先清理一下可能存在的Excel进程
    pids = psutil.pids()
    for pid in pids:
        try:
            p = psutil.Process(pid)
            # print('pid=%s,pname=%s' % (pid, p.name()))
            # 关闭excel进程
            if p.name() == 'EXCEL.EXE':
                cmd = 'taskkill /F /IM EXCEL.EXE'
                os.system(cmd)
        except Exception as e:
            print(e)


def refresh_multi_pivottable():
    App = xw.App(visible=False, add_book=False)
    wb = App.books.open('数据透视表_多个透视表.xlsx')
    sheet = wb.sheets('数据透视表')
    print('sheet:', sheet)
    # 获取透视表的个数
    num = sheet.api.PivotTables().Count
    # 循环进行透视表的刷新
    for i in range(1, num+1):
        sheet.api.PivotTables(i).PivotCache().Refresh()
    # 保存刷新的结果
    wb.save()
    wb.close()
    App.quit()


kill_excel_by_pid()
refresh_multi_pivottable()
kill_excel_by_pid()