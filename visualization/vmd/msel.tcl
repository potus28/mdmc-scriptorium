# vmd msel.pdb -e msel.tcl
# the run command in terminal


logfile msel.out
# creates a logfile that shows every command run

# Commands for making a new material

# material add <material name>
# material change <parameter> <material name> <value>

material add MSEL
material change ambient MSEL 0
material change diffuse MSEL 0.9
material change specular MSEL 0.2
material change shininess MSEL 0.53
material change mirror MSEL 0
material change opacity MSEL 1
material change outline MSEL 4
material change outlinewidth MSEL 0.5

# Change orientation commands
#rotate z by 90
#rotate x by 90
#rotate z by 180
#rotate y by -30
#rotate x by 15

color Display Background white
# change the background color

display depthcue on
display shadows on
display ambientocclusion on

display projection orthographic
# make perspective orthographic

# Commands for adding representations

#mol delrep 0 0
# specify molecule id and then representation id

#mol addrep 0
# specify molecule id to add the new representation to, will append to existing representations

#mol modselect 0 0 all
# modifies which atoms a specific representation represents, again specify molecule id and then representation id

#mol modmaterial 0 0 MSEL
# modifies material used

mol modstyle 0 0 CPK 1.300000 0.600000 30.000000 30.000000
mol modstyle 0 0 vdw 0.6 30
# modifies drawing method, parameters for the drawing method are specified after the drawing method
# here, it is sphere size and sphere resolution

mol modcolor 0 0 element
# modifies coloring method

#pbc box -style tubes -width 1 -material MSEL -resolution 30 -color gray
# draws the pbc boundary box

axes location off
# removes the axes before the render

#render Tachyon water_example.dat '/Applications/VMD 1.9.4a57-x86_64-Rev12.app/Contents/vmd/tachyon_MACOSXX86_64' -aasamples 36 -res 2048 2048 -add_skylight 1.0 %s -format TARGA -o %s.tga
# the filename in the render command is for the output file

