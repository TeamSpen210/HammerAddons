// Created by Crowbar 0.55

$ModelName "props_backstage/vacum_scanner_tv_vert_new.mdl"

$staticprop

$BodyGroup "default"
{
	studio "vacum_scanner_tv_vertical_model.smd"
}

$SurfaceProp "glass"

$Contents "solid"

$IllumPosition -14.676 -36.82 3.817

$CDMaterials "models\props_backstage\"

$TextureGroup "skinfamilies"
{
	{ "vacum_scanner_tv_vert_blank"       }
	{ "vacum_scanner_tv_vert_chair"       }
	{ "vacum_scanner_tv_vert_table"       }
	{ "vacum_scanner_tv_vert_box"         }
	{ "vacum_scanner_tv_vert_hover"       }
	{ "vacum_scanner_tv_vert_turret01"    }
	{ "vacum_scanner_tv_vert_boxedturret" }
}

$CBox 0 0 0 0 0 0

$BBox 0 -35.964 -12.413 73.64 6.611 20.048

$Sequence "BindPose" {
	"vacum_scanner_tv_vertical_anims\BindPose.smd"
	fadein 0.2
	fadeout 0.2
	fps 30
}
