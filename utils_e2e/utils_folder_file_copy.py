import os
import shutil


from .utils_copy_files import copyFile


def unique_spk_list(spk_list):
    """To compute a list of unique speakers from ``spk_list''."""
    spk_list_unique = []
    for i in spk_list:
        if i not in spk_list_unique:
            spk_list_unique.append(i)

    return spk_list_unique


def create_spks_per_agnt_dataset(root_dir, dest_dir_agnt, agnt_indx, num_spk_per_agnt):
    """To copy ``num_spk_per_agnt:int'' folders of speakers and their contents
    from the ``root_dir'' to ``dest_dir_agnt'' for agent ``agnt_indx:int''."""

    spk_list = []
    for path, folder, files in os.walk(root_dir):
        for file in files:
            src_path = os.path.join(path, file)
            src_path_list = src_path.split("\\")

            if len(src_path_list) > 3:
                spk_list.append(src_path_list[3])

            spk_list_unique = unique_spk_list(spk_list)
            spk_list_unique_slice = spk_list_unique[agnt_indx * num_spk_per_agnt :]

            if (
                len(spk_list_unique_slice) <= num_spk_per_agnt
                and len(spk_list_unique_slice) != 0
            ):

                if len(src_path_list) > 3:

                    # print(len(spk_list_unique_slice) / num_spk_per_agnt)

                    src_path_sub_folder = "\\".join(src_path_list[3:-1])

                    new_dest_dir = os.path.join(dest_dir_agnt, src_path_sub_folder)
                    if not os.path.exists(new_dest_dir):
                        os.makedirs(new_dest_dir)
                    dst_path = os.path.join(new_dest_dir, file)

                    shutil.copy(src_path, dst_path)
                    # copyFile(src_path, dst_path)

            elif len(src_path_list) <= 3:

                if not os.path.exists(dest_dir_agnt):
                    os.makedirs(dest_dir_agnt)
                dst_path = os.path.join(dest_dir_agnt, file)

                shutil.copy(src_path, dst_path)
                # copyFile(src_path, dst_path)

            else:
                break
