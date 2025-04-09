import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_objects_by_rects(image_path, method=cv2.INPAINT_TELEA, inpaint_radius=3):
    """
    画像からユーザーが指定した複数の矩形領域の物体を除去。

    Parameters:
        image_path (str): 入力画像のパス
        method (int): inpaint の手法（cv2.INPAINT_TELEA または cv2.INPAINT_NS）
        inpaint_radius (int): inpaint の影響範囲（初期値 3）

    Returns:
        dst (np.ndarray): 除去後の画像（BGR）
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    click_points = []
    rects = []

    def on_click(event):
        if event.xdata is not None and event.ydata is not None and event.button == 1:
            point = (int(event.xdata), int(event.ydata))
            click_points.append(point)
            print(f"クリック: {point}")
            if len(click_points) % 2 == 0:
                pt1 = click_points[-2]
                pt2 = click_points[-1]
                rects.append((pt1, pt2))
                ax.add_patch(plt.Rectangle(pt1, pt2[0] - pt1[0], pt2[1] - pt1[1],
                                           fill=False, edgecolor='red', linewidth=2))
                fig.canvas.draw()

    def on_key(event):
        if event.key == 'enter':
            print("Enter が押されたので除去を開始します")
            plt.close()

    # 表示とイベント登録
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("矩形を複数選択（左上→右下をペアでクリック）。Enterで完了")
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # 矩形からマスクを作成
    for (pt1, pt2) in rects:
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 除去
    dst = cv2.inpaint(img, mask, inpaint_radius, method)

    # 結果表示
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title("除去後の画像")
    plt.axis('off')
    plt.show()

    return dst

result = remove_objects_by_rects('img/target.png')
cv2.imwrite('opencv9_result.jpg', result)

