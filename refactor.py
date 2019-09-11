from parse_utils import PeTrackParser

scn_nbr = 9
run_nbr = 3
parser = PeTrackParser()
main_dir = 'C:/Users/Javad/Dropbox/PAMELA data/new_cut_video'

p_heads, t_head, ids = parser.load(main_dir + '/S%d/run%d/S%d_run%d-heads.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
p_legs, t_leg, _ = parser.load(main_dir + '/S%d/run%d/S%d_run%d-legs.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr))
t_data = t_head
output_filename = main_dir + '/S%d/run%d/S%d_run%d.txt' % (scn_nbr, run_nbr, scn_nbr, run_nbr)

assert len(p_legs) == len(p_heads), "mismatch - heads and legs"
assert len(t_data) == len(p_legs), "mismatch - times and locs"
for ii, Ti in enumerate(t_data):
    assert len(Ti) == len(p_heads[ii]), "mismatch - heads - %d" % ii
    assert len(Ti) == len(p_legs[ii]), "mismatch - legs - %d" % ii


with open(output_filename, 'w') as out_file:
    out_file.write('# id frame foot_x/m foot_y/m foot_z/m head_x/m head_y/m head_z/m\n')
    robot_id = -1 if scn_nbr < 3 else 1
    out_file.write('# ids = %d\n' % len(t_data))
    out_file.write('# Robot id = %d\n' % robot_id)

    for ii, Ti in enumerate(t_data):
        for kk, t in enumerate(Ti):
            out_file.write("%d %d %.3f %.3f %.2f %.3f %.3f %.2f\n"
                           % (ii + 1, t,
                              p_legs[ii][kk, 0] / 100, p_legs[ii][kk, 1] / -100, 0.,
                              p_heads[ii][kk, 0] / 100, p_heads[ii][kk, 1] / -100, 1.70))

