import detectorlib

path="2022_LOU_AZE_field_maturation_final_matrix.xlsx"
path2='2021_All raw data_No_background.xlsx'
graph=detectorlib.printer()
graph.load(path2, 6)

graph.print_all()