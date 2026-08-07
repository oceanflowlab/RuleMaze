# import numpy as np
# import cv2


def find_continuous_max_region(arr, value, threshold):
    max_len = 0
    current_len = 0
    start_index = -1
    temp_start = -1

    for i in range(len(arr)):
        if arr[i] == value:
            if current_len == 0:
                temp_start = i
            current_len += 1
        else:
            if current_len > max_len:
                max_len = current_len
                start_index = temp_start
            current_len = 0

    # Final check at the end of the array
    if current_len > max_len:
        max_len = current_len
        start_index = temp_start

    if max_len >= threshold:
        return start_index, max_len
    else:
        return -1, 0


def find_one_lag(dark, img_gray, horizontal_=True, start_row=0, line_width=0):
    H, W = img_gray.shape
    if not horizontal_:
        dark = dark.T
        H, W = W, H
    # 1. Starting from start_row, find the first y0 that satisfies the condition.
    y0 = -1
    for h in range(start_row, H):
        row = dark[h, :]
        lag_x_ = find_continuous_max_region(
            row, 255, threshold=(W // 2) if line_width == 0 else line_width * 2
        )[0]
        if lag_x_ != -1:
            y0 = h
            break
    if y0 == -1:
        # Nothing remains after this point.
        return None, H  # lag_x=None, the next call can stop immediately.

    # 2. Move downward to find the end of the continuous region and compute line_width.
    mark = -1
    for h in range(y0, H):
        row = dark[h, :]
        lag_x_ = find_continuous_max_region(
            row, 255, threshold=(W // 2) if line_width == 0 else line_width * 2
        )[0]
        if lag_x_ == -1:
            line_width = h - y0
            mark = h
            break
    if mark == -1:
        # The continuous region reaches the bottom, so lag_x cannot be computed.
        return None, H

    # 3. From mark, find the next stronger match and compute lag_x.
    for h in range(mark, H):
        row = dark[h, :]
        lag_x_ = find_continuous_max_region(
            row, 255, threshold=line_width * 2
        )[0]
        if lag_x_ != -1:
            lag_x = h - mark
            # Return lag_x and tell the caller which row to continue from next.
            return lag_x, h + 1, y0, line_width

    # No second segment found.
    return None, H
