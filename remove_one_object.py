import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_object_by_rect(image_path, method=cv2.INPAINT_TELEA, inpaint_radius=3):
    """
    入力画像を読み込み、ユーザーが指定した矩形領域の物体を除去する。

    Parameters:
        image_path (str): 入力画像のパス
        method (int): cv2.INPAINT_TELEA または cv2.INPAINT_NS（デフォルト: TELEA）
        inpaint_radius (int): 除去の範囲 (デフォルト: 3)

    Returns:
        dst (np.ndarray): 除去後の画像（BGR形式）
    """

    # 入力画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    coords = []

    # クリックイベント
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            coords.append((int(event.xdata), int(event.ydata)))
            print(f"点 {len(coords)}: {coords[-1]}")
            if len(coords) == 2:
                plt.close()

    # 画像表示してクリックを待つ
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Click top-left and bottom-right corners")
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    # 2点選ばれたか確認
    if len(coords) != 2:
        raise ValueError("矩形を指定するために2点クリックしてください")

    pt1, pt2 = coords
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])

    # マスク生成
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 除去処理
    dst = cv2.inpaint(img, mask, inpaint_radius, method)

    # 結果表示
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title("Result (object removed)")
    plt.axis('off')
    plt.show()

    return dst


result = remove_object_by_rect('img/target.png')
cv2.imwrite('opencv9_result.jpg', result)
