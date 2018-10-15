#!/bin/bash
FILEDIR=$1
cd $FILEDIR
for f in $FILEDIR/*
do
  cd $FILEDIR
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  FILENAME=${f##/*/}
  SHORTNAME=${FILENAME%_google_16k.tgz}

  tar -xf $FILENAME --exclude nontextured.stl  --exclude nontextured.stl  --exclude nontextured.ply  --exclude textured.obj  --exclude textured.mtl  --exclude kinbody.xml

  cd $SHORTNAME

  mkdir materials

  mkdir materials/scripts
  mkdir materials/textures


  cat > model.config <<EOF
  <?xml version="1.0"?>

  <model>
    <name>$SHORTNAME</name>
    <version>1.0</version>
    <sdf version="1.5">model.sdf</sdf>

    <author>
        <name>Markus Ernst</name>
        </author>
    <description>
    Model for $SHORTNAME
  </description>
  </model>
EOF



cat > model.sdf <<EOF
<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="$SHORTNAME">
    <link name="$SHORTNAME link">
<!--
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>0.0</mass>
        <inertia>
          <ixx>0</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0</iyy>
          <iyz>0</iyz>
          <izz>0</izz>
        </inertia>
      </inertial>
      <collision name="$SHORTNAME collision">
        <geometry>
           <mesh>
<uri>model://$SHORTNAME/textured.dae</uri>
</mesh>
        </geometry>
      </collision>
-->
      <visual name="$SHORTNAME visual">
        <geometry>
          <mesh>
<uri>model://$SHORTNAME/textured.dae</uri>
</mesh>
        </geometry>
        <material>
          <script>
            <uri>model://$SHORTNAME/materials/scripts</uri>
            <uri>model://$SHORTNAME/materials/textures</uri>
            <name>$SHORTNAME</name>
          </script>
        </material>
      </visual>
	  <gravity> 0 </gravity>
    </link>
  </model>
</sdf>
EOF

cat > materials/scripts/$SHORTNAME.material <<EOF
material $SHORTNAME
{
  technique
  {
    pass
    {
      texture_unit
      {
        texture $SHORTNAME.png
        filtering anistropic
        max_anisotropy 16
      }
    }
  }
}

EOF



mv google_16k/* ../$SHORTNAME
rm -r google_16k

mv texture_map.png materials/textures/$SHORTNAME.png
rm $f
done
