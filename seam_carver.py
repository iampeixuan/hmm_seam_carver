import cv2
import numpy as np


class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        # read in image
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)
        self.out_height = out_height
        self.out_width = out_width

        self.seams_carving()

    def seams_carving(self):
        # determine how many seams to remove
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        if delta_row > 0 or delta_col > 0:
            print("Does not support image enlarging, exiting")
            exit(0)

        if delta_row != 0:
            print("Only supports horizontal resizing for now, exiting...")
            exit(0)

        if delta_col < 0:
            for dummy in range(delta_col * -1):
                energy_map = self.calc_energy_map()
                seam_idx = self.viterbi(energy_map)
                self.delete_seam(seam_idx)

    def calc_energy_map(self):
        # energy is calculated as the squared component-wise distance between 4 neighbours of the pixel
        b, g, r = cv2.split(self.out_image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy

    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)

    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))

    def viterbi(self, energy_map):
        num_rows, num_cols = energy_map.shape

        # used as parent pointer, starting from 2nd row
        pixel_index = np.zeros((num_rows - 1, num_cols))

        # record the sum of min energy pixels in the seam up to this row
        current_seam_energy = energy_map[0]

        for y in range(1, num_rows):
            new_seam_energy = []

            # go through every pixel in the row, find the min energy from the neighbours of current_seam_energy
            for x, pixel_energy in enumerate(energy_map[y]):
                # if pixel is at the left-most
                if x == 0:
                    neighbour_energies = [current_seam_energy[0], current_seam_energy[1]]
                    min_energy = min(neighbour_energies)
                    min_index = np.argmin(neighbour_energies)
                
                # if pixel is at the right-most
                elif x == num_cols - 1:
                    neighbour_energies = [current_seam_energy[num_cols - 2], current_seam_energy[num_cols - 1]]
                    min_energy = min(neighbour_energies)
                    temp = np.argmin(neighbour_energies)
                    if temp == 1:
                        min_index = num_cols - 1
                    else:
                        min_index = num_cols - 2
                
                # otherwise the pixel has 3 neighbours
                else:
                    neighbour_energies = [current_seam_energy[x - 1], current_seam_energy[x], current_seam_energy[x + 1]]
                    min_energy = min(neighbour_energies)
                    temp = np.argmin(neighbour_energies)
                    if temp == 0:
                        min_index = x - 1
                    elif temp == 1:
                        min_index = x
                    else:
                        min_index = x + 1

                # record the index (where the min erengy come from) and add to the seam energy
                pixel_index[y-1][x] = int(min_index)
                new_seam_energy.append(pixel_energy + min_energy)

            # update seam energy, for the next row to use
            current_seam_energy = new_seam_energy

        # backtrack from the last row to find the overall lowest energy seam
        min_idx = int(np.argmin(current_seam_energy))
        seam_to_remove = [min_idx]
        for y in range(num_rows - 2, -1, -1):
            min_idx = int(pixel_index[y][min_idx])
            seam_to_remove.append(min_idx)

        # restore the top-down order
        seam_to_remove.reverse()
        #print(seam_to_remove[0])

        return seam_to_remove
