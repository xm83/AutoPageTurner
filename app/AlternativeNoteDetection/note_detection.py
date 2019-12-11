import subprocess
import cv2
import time
import numpy as np
from random import randint
import os
import sys

from .best_fit import fit
from .rectangle import Rectangle
from .note import Note

staff_files = [
    "/AlternativeNoteDetection/resources/template/staff10.png",
    "/AlternativeNoteDetection/resources/template/staff.png",
    ]
quarter_files = [
    "/AlternativeNoteDetection/resources/template/quarter.png", 
    "/AlternativeNoteDetection/resources/template/solid-note.png"]
sharp_files = [
    "/AlternativeNoteDetection/resources/template/sharp.png",
    "/AlternativeNoteDetection/resources/template/f-sharp.png",
    ]
flat_files = [
    "/AlternativeNoteDetection/resources/template/flat-line.png", 
    "/AlternativeNoteDetection/resources/template/flat-space.png" ]
half_files = [
    "/AlternativeNoteDetection/resources/template/half-space.png", 
    "/AlternativeNoteDetection/resources/template/half-note-line.png",
    "/AlternativeNoteDetection/resources/template/half-line.png", 
    "/AlternativeNoteDetection/resources/template/half-note-space.png",
    "/AlternativeNoteDetection/resources/template/half-line2.png"]
whole_files = [
    "/AlternativeNoteDetection/resources/template/whole-space.png", 
    "/AlternativeNoteDetection/resources/template/whole-note-line.png",
    "/AlternativeNoteDetection/resources/template/whole-line.png", 
    "/AlternativeNoteDetection/resources/template/whole-note-space.png"]

staff_imgs = [cv2.imread(os.getcwd() + staff_file, 0) for staff_file in staff_files]
quarter_imgs = [cv2.imread(os.getcwd() + quarter_file, 0) for quarter_file in quarter_files]
sharp_imgs = [cv2.imread(os.getcwd() + sharp_files, 0) for sharp_files in sharp_files]
flat_imgs = [cv2.imread(os.getcwd() + flat_file, 0) for flat_file in flat_files]
half_imgs = [cv2.imread(os.getcwd() + half_file, 0) for half_file in half_files]
whole_imgs = [cv2.imread(os.getcwd() + whole_file, 0) for whole_file in whole_files]

staff_lower, staff_upper, staff_thresh = 50, 250, 0.75
sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
flat_lower, flat_upper, flat_thresh = 50, 150, 0.77
# quarter_lower, quarter_upper, quarter_thresh = 50, 150, 0.70
quarter_lower, quarter_upper, quarter_thresh = 50, 250, 0.70
half_lower, half_upper, half_thresh = 50, 150, 0.70
whole_lower, whole_upper, whole_thresh = 50, 150, 0.70

def is_not_empty(structure1):
    if structure1 and len(list(structure1)) > 0:
        return True
    else:
        return False

def locate_images(img, templates, start, stop, threshold):
    locations, scale = fit(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        
        img_locations.append([Rectangle(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
          
    return img_locations

def merge_recs(recs, threshold):
    filtered_recs = []
    
    
    while len(recs) > 0:
        r = recs.pop(0)
        recs.sort(key=lambda rec: rec.distance(r))
        merged = True
        while(merged):
            merged = False
            i = 0
            for _ in range(len(recs)):
                if r.overlap(recs[i]) > threshold or recs[i].overlap(r) > threshold:
                    r = r.merge(recs.pop(i))
                    merged = True
                elif recs[i].distance(r) > r.w/2 + recs[i].w/2:
                    break
                else:
                    i += 1
        filtered_recs.append(r)

    return filtered_recs


def open_file(path):
    # img = Image.open(path)
    # img.show()
    cmd = {'linux':'eog', 'win32':'explorer', 'darwin':'open'}[sys.platform]
    subprocess.run([cmd, path])

def note_detection(img):
    # img_file = sys.argv[1:][0]
    # img = cv2.imread(img_file, 0)
    print("Got img in note_detection")

    # ============ Noise Removal ============
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # # ============ Binarization ============
    
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's Thresholding
    # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_width, img_height = img.shape[::-1]

    # print(staff_imgs)
    
    staff_recs = locate_images(img, staff_imgs, staff_lower, staff_upper, staff_thresh)

    
    print("Filtering weak staff matches...")
    staff_recs = [j for i in staff_recs for j in i]
    heights = [r.y for r in staff_recs] + [0]
    histo = [heights.count(i) for i in range(0, max(heights) + 1)]
    avg = np.mean(list(set(histo)))
    staff_recs = [r for r in staff_recs if histo[r.y] > avg]

    print("Merging staff image results...")
    staff_recs = merge_recs(staff_recs, 0.01)
    staff_recs_img = img.copy()
    for r in staff_recs:
        r.draw(staff_recs_img, (0, 0, 255), 2)
    # cv2.imwrite('staff_recs_img.png', staff_recs_img)
    # open_file('staff_recs_img.png')

    print("Discovering staff locations...")
    staff_boxes = merge_recs([Rectangle(0, r.y, img_width, r.h) for r in staff_recs], 0.01)
    staff_boxes_img = img.copy()
    for r in staff_boxes:
        r.draw(staff_boxes_img, (0, 0, 255), 2)


    print("Matching sharp image...")
    sharp_recs = locate_images(img, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)

    print("Merging sharp image results...")
    sharp_recs = merge_recs([j for i in sharp_recs for j in i], 0.5)
    sharp_recs_img = img.copy()
    for r in sharp_recs:
        r.draw(sharp_recs_img, (0, 0, 255), 2)
    # cv2.imwrite('sharp_recs_img.png', sharp_recs_img)
    # open_file('sharp_recs_img.png')

    print("Matching flat image...")
    flat_recs = locate_images(img, flat_imgs, flat_lower, flat_upper, flat_thresh)

    print("Merging flat image results...")
    flat_recs = merge_recs([j for i in flat_recs for j in i], 0.5)
    flat_recs_img = img.copy()
    for r in flat_recs:
        r.draw(flat_recs_img, (0, 0, 255), 2)
    # cv2.imwrite('flat_recs_img.png', flat_recs_img)
    # open_file('flat_recs_img.png')

    print("Matching quarter image...")
    quarter_recs = locate_images(img, quarter_imgs, quarter_lower, quarter_upper, quarter_thresh)

    print("Merging quarter image results...")
    quarter_recs = merge_recs([j for i in quarter_recs for j in i], 0.5)
    quarter_recs_img = img.copy()
    for r in quarter_recs:
        r.draw(quarter_recs_img, (0, 0, 255), 2)
    # cv2.imwrite('quarter_recs_img.png', quarter_recs_img)
    # open_file('quarter_recs_img.png')

    print("Matching half image...")
    half_recs = locate_images(img, half_imgs, half_lower, half_upper, half_thresh)

    print("Merging half image results...")
    half_recs = merge_recs([j for i in half_recs for j in i], 0.5)
    half_recs_img = img.copy()
    for r in half_recs:
        r.draw(half_recs_img, (0, 0, 255), 2)
    # cv2.imwrite('half_recs_img.png', half_recs_img)
    # open_file('half_recs_img.png')

    print("Matching whole image...")
    whole_recs = locate_images(img, whole_imgs, whole_lower, whole_upper, whole_thresh)

    print("Merging whole image results...")
    whole_recs = merge_recs([j for i in whole_recs for j in i], 0.5)
    whole_recs_img = img.copy()
    for r in whole_recs:
        r.draw(whole_recs_img, (0, 0, 255), 2)
    # cv2.imwrite('whole_recs_img.png', whole_recs_img)
    # open_file('whole_recs_img.png')

    print("Combining results")

    note_groups = []
    for box in staff_boxes:
        staff_sharps = [Note(r, "sharp", box) 
            for r in sharp_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        staff_flats = [Note(r, "flat", box) 
            for r in flat_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        quarter_notes = [Note(r, "4,8", box, staff_sharps, staff_flats) 
            for r in quarter_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        half_notes = [Note(r, "2", box, staff_sharps, staff_flats) 
            for r in half_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        whole_notes = [Note(r, "1", box, staff_sharps, staff_flats) 
            for r in whole_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        staff_notes = quarter_notes + half_notes + whole_notes
        staff_notes.sort(key=lambda n: n.rec.x)
        staffs = [r for r in staff_recs if r.overlap(box) > 0]
        staffs.sort(key=lambda r: r.x)
        note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        note_group = []
        i = 0; j = 0;

        print("len(staff_notes)", len(staff_notes))
        print("len(staffs)", len(staffs))
        while(i < len(staff_notes)):
            # if (staff_notes[i].rec.x > staffs[j].x and j < len(staffs)):
            if j < len(staffs):
                print("staff_notes[i].rec.x", staff_notes[i].rec.x)
                print("staffs[j].x", staffs[j].x)
                if staff_notes[i].rec.x > staffs[j].x:
                    r = staffs[j]
                    j += 1;
                    print(len(note_group))
                    if len(note_group) > 0:
                        note_groups.append(note_group)
                        note_group = []
                    note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            else:
                note_group.append(staff_notes[i])
                staff_notes[i].rec.draw(img, note_color, 2)
                i += 1
        note_groups.append(note_group)
        print(len(note_group))

    for r in staff_boxes:
        r.draw(img, (0, 0, 255), 2)
    for r in sharp_recs:
        r.draw(img, (0, 0, 255), 2)
    flat_recs_img = img.copy()
    for r in flat_recs:
        r.draw(img, (0, 0, 255), 2)
        
    cv2.imwrite('res.png', img)
    # open_file('res.png')
   
    notes_array = []
    for note_group in note_groups:
        duration = None
        for note in note_group:
            if note.sym == "1":
                duration = 4
            elif note.sym == "2":
                duration = 2
            elif note.sym == "4,8":
                duration = 1
                print("length of note group", len(note_group))
                # duration = 1 if len(note_group) == 1 else 0.5
            print(note.note + ", duration:" + str(duration))

            for i in range(int(4*duration)):
                notes_array.append(note.note)
    return notes_array
    