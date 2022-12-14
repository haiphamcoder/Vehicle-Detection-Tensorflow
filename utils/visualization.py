import collections
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy
from utils.speed_and_direction_prediction_module import speed_prediction
from utils.color_recognition_module import color_recognition_api


class Visualization:
    is_vehicle_detected = [0]
    ROI_POSITION = 200
    STANDARD_COLORS = ['AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'BlanchedAlmond',
                       'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue',
                       'Cornsilk', 'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
                       'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
                       'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia',
                       'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink',
                       'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon',
                       'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
                       'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray',
                       'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
                       'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue',
                       'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
                       'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod',
                       'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum',
                       'PowderBlue', 'Purple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
                       'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey',
                       'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise',
                       'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen']

    def __init__(self):
        pass

    def visualize_boxes_and_labels_on_image_array(self, current_frame_number, image, boxes, classes, scores,
                                                  category_index, instance_masks=None, keypoints=None,
                                                  use_normalized_coordinates=False, max_boxes_to_draw=20,
                                                  min_score_thresh=.5, agnostic_mode=False, line_thickness=4):
        """Overlay labeled boxes on an image with formatted scores and label names.

        This function groups boxes that correspond to the same location
        and creates a display string for each detection and overlays these
        on the image. Note that this function modifies the image in place, and returns
        that same image.

        Args:
        image: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
        instance_masks: a numpy array of shape [N, image_height, image_width], can
            be None
        keypoints: a numpy array of shape [N, num_keypoints, 2], can
            be None
        use_normalized_coordinates: whether boxes is to be interpreted as
            normalized coordinates or not.
        max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
            all boxes.
        min_score_thresh: minimum score threshold for a box to be visualized
        agnostic_mode: boolean (default: False) controlling whether to evaluate in
            class-agnostic mode or not.  This mode will display scores but ignore
            classes.
        line_thickness: integer (default: 4) controlling line width of the boxes.

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
        """
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        csv_line_util = "not_available"
        counter = 0
        is_vehicle_detected = []
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_keypoints_map = collections.defaultdict(list)
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if keypoints is not None:
                    box_to_keypoints_map[box].extend(keypoints[i])
                if scores is None:
                    box_to_color_map[box] = 'black'
                else:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))
                    else:
                        display_str = 'score: {}%'.format(int(100 * scores[i]))

                    box_to_display_str_map[box].append(display_str)
                    if agnostic_mode:
                        box_to_color_map[box] = 'DarkOrange'
                    else:
                        box_to_color_map[box] = self.STANDARD_COLORS[classes[i] % len(self.STANDARD_COLORS)]

        # Draw all boxes onto image.
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            if instance_masks is not None:
                self.draw_mask_on_image_array(image, box_to_instance_masks_map[box], color=color)

            display_str_list = box_to_display_str_map[box]
            # we are interested just vehicles (i.e. cars and trucks)
            if ("car" in display_str_list[0]) or ("truck" in display_str_list[0]) or ("bus" in display_str_list[0]):
                is_vehicle_detected, csv_line, update_csv = self.draw_bounding_box_on_image_array(current_frame_number,
                                                                                                  image, ymin, xmin,
                                                                                                  ymax, xmax,
                                                                                                  color=color,
                                                                                                  thickness=line_thickness,
                                                                                                  display_str_list=
                                                                                                  box_to_display_str_map[
                                                                                                      box],
                                                                                                  use_normalized_coordinates=use_normalized_coordinates)

                if keypoints is not None:
                    self.draw_keypoints_on_image_array(image, box_to_keypoints_map[box], color=color,
                                                       radius=line_thickness / 2,
                                                       use_normalized_coordinates=use_normalized_coordinates)

        if 1 in is_vehicle_detected:
            counter = 1
            del is_vehicle_detected[:]
            is_vehicle_detected = []
            if class_name == "boat":
                class_name = "truck"
            csv_line_util = class_name + "," + csv_line

        return counter, csv_line_util

    def draw_mask_on_image_array(self, image, mask, color='red', alpha=0.7):
        """Draws mask on an image.

        Args:
        image: uint8 numpy array with shape (img_height, img_height, 3)
        mask: a uint8 numpy array of shape (img_height, img_height) with
            values between either 0 or 1.
        color: color to draw the keypoints with. Default is red.
        alpha: transparency value between 0 and 1. (default: 0.7)

        Raises:
        ValueError: On incorrect data type for image or masks.
        """
        if image.dtype != np.uint8:
            raise ValueError('`image` not of type np.uint8')
        if mask.dtype != np.uint8:
            raise ValueError('`mask` not of type np.uint8')
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        rgb = ImageColor.getrgb(color)
        pil_image = Image.fromarray(image)

        solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))

    def draw_bounding_box_on_image_array(self, current_frame_number, image, ymin, xmin, ymax, xmax, color='red',
                                         thickness=4, display_str_list=(), use_normalized_coordinates=True):
        """Adds a bounding box to an image (numpy array).

        Args:
        image: a numpy array with shape [height, width, 3].
        ymin: ymin of bounding box in normalized coordinates (same below).
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                            (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
            coordinates as absolute.
        """
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        is_vehicle_detected, csv_line, update_csv = self.draw_bounding_box_on_image(current_frame_number, image_pil,
                                                                                    ymin, xmin, ymax, xmax, color,
                                                                                    thickness, display_str_list,
                                                                                    use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))
        return is_vehicle_detected, csv_line, update_csv

    def draw_keypoints_on_image_array(self, image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
        """Draws keypoints on an image (numpy array).

        Args:
        image: a numpy array with shape [height, width, 3].
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        radius: keypoint radius. Default value is 2.
        use_normalized_coordinates: if True (default), treat keypoint values as
            relative to the image.  Otherwise treat them as absolute.
        """
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        self.draw_keypoints_on_image(image_pil, keypoints, color, radius, use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))

    def draw_keypoints_on_image(self, image, keypoints, color='red', radius=2, use_normalized_coordinates=True):
        """Draws keypoints on an image.

        Args:
        image: a PIL.Image object.
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        radius: keypoint radius. Default value is 2.
        use_normalized_coordinates: if True (default), treat keypoint values as
            relative to the image.  Otherwise treat them as absolute.
        """
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        keypoints_x = [k[1] for k in keypoints]
        keypoints_y = [k[0] for k in keypoints]
        if use_normalized_coordinates:
            keypoints_x = tuple([im_width * x for x in keypoints_x])
            keypoints_y = tuple([im_height * y for y in keypoints_y])
        for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
            draw.ellipse([(keypoint_x - radius, keypoint_y - radius), (keypoint_x + radius, keypoint_y + radius)],
                         outline=color, fill=color)

    def draw_bounding_box_on_image(self, current_frame_number, image, ymin, xmin, ymax, xmax, color='red', thickness=4,
                                   display_str_list=(), use_normalized_coordinates=True):
        """Adds a bounding box to an image.

        Each string in display_str_list is displayed on a separate line above the
        bounding box in black text on a rectangle filled with the input 'color'.
        If the top of the bounding box extends to the edge of the image, the strings
        are displayed below the bounding box.

        Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                            (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
            coordinates as absolute.
        """
        image_temp = numpy.array(image)
        csv_line = ""  # to create new csv line consists of vehicle type, predicted_speed, color and predicted_direction
        # update csv for a new vehicle that are passed from ROI - just one new line for each vehicles
        update_csv = False
        is_vehicle_detected = [0]
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness,
                  fill=color)

        predicted_speed = "n.a."  # means not available, it is just initialization
        predicted_direction = "n.a."  # means not available, it is just initialization

        detected_vehicle_image = image_temp[int(top):int(bottom), int(left):int(right)]

        if bottom > self.ROI_POSITION:  # if the vehicle get in ROI area, vehicle predicted_speed
            # predicted_color algorithms are called - 200 is an arbitrary value, for my case it looks very well to
            # set position of ROI line at y pixel 200
            predicted_direction, predicted_speed, is_vehicle_detected, update_csv = speed_prediction.predict_speed(top,
                                                                                                                   bottom,
                                                                                                                   right,
                                                                                                                   left,
                                                                                                                   current_frame_number,
                                                                                                                   detected_vehicle_image,
                                                                                                                   self.ROI_POSITION)

        predicted_color = color_recognition_api.color_recognition(detected_vehicle_image)

        try:
            font = ImageFont.truetype('arial.ttf', 16)
        except IOError:
            font = ImageFont.load_default()

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_list[0] = predicted_color + " " + display_str_list[0]
        csv_line = predicted_color + "," + str(predicted_direction) + "," + str(predicted_speed)  # csv line created
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        # Reverse list and print from bottom to top.
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
                           fill=color)
            draw.text((left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
            text_bottom -= text_height - 2 * margin
            return is_vehicle_detected, csv_line, update_csv
