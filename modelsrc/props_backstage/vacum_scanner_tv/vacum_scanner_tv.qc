// Created by Crowbar 0.55

$ModelName "props_backstage/vacum_scanner_tv_horiz_new.mdl"

$StaticProp

$BodyGroup "default"
{
	studio "vacum_scanner_tv_model.smd"
}

$SurfaceProp "glass"

$Contents "solid"

$IllumPosition -14.676 -36.82 3.817

$CDMaterials "models\props_backstage"

$TextureGroup "skinfamilies"
{
	{ "vacum_scanner_tv_blank01"     }
	{ "vacum_scanner_tv_chair01"     }
	{ "vacum_scanner_tv_table01"     }
	{ "vacum_scanner_tv_cube01"      }
	{ "vacum_scanner_tv_hover01"     }
	{ "vacum_scanner_tv_turret01"    }
	{ "vacum_scanner_tv_boxedturret" }
}

$CBox 0 0 0 0 0 0

$BBox 0 -35.964 -12.413 73.64 6.611 20.048

$Sequence "BindPose" {
	"vacum_scanner_tv_anims\BindPose.smd"
	fadein 0.2
	fadeout 0.2
	fps 30
}
