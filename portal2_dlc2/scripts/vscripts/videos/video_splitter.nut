// --------------------------------------------------------
// StartVideo
// --------------------------------------------------------

RandomVideos <-
[
	"animalking.bik",
	"aperture_appear_horiz.bik",
	"bluescreen.bik",
	"coop_bluebot_load.bik",
	"coop_bots_load.bik",
	"coop_bots_load_wave.bik",
	"coop_orangebot_load.bik",
	"exercises_horiz.bik",
	"faithplate.bik",
	"fizzler.bik",
	"hard_light.bik",
	"laser_danger_horiz.bik",
	"laser_portal.bik",
	"plc_blue_horiz.bik",
	"turret_colours_type.bik",
	"turret_dropin.bik",
	"turret_exploded_grey.bik",
	"community_bg1.bik"
]

ElevatorVideos <- 
[
	{ map = "sp_a1_intro1", arrival = "", departure = "" },
	{ map = "sp_a1_intro2", arrival = "", departure = "" },
	{ map = "sp_a1_intro3", arrival = "animalking.bik", departure = "animalking.bik", typeOverride = 11  },
	{ map = "sp_a1_intro4", arrival = "exercises_horiz.bik", departure = "exercises_horiz.bik", typeOverride = 10 },
	{ map = "sp_a1_intro5", arrival = "exercises_vert.bik", departure = "exercises_vert.bik", typeOverride = 9 },
	{ map = "sp_a1_intro6", arrival = "plc_blue_vert.bik", departure = "plc_blue_vert.bik", typeOverride = 9 },
	{ map = "sp_a1_intro7", arrival = "plc_blue_horiz.bik", departure = "", typeOverride = 4 },
	{ map = "sp_a2_intro", arrival = "", departure = "plc_blue_horiz.bik", typeOverride = 1 },
	{ map = "sp_a2_laser_intro",	arrival = "laser_portal.bik", departure = "laser_portal.bik", typeOverride = 12  },
	{ map = "sp_a2_laser_stairs",	arrival = "laser_portal.bik", departure = "laser_portal.bik", typeOverride = 12 },
	{ map = "sp_a2_dual_lasers",	arrival = "laser_portal.bik", departure = "laser_portal.bik", typeOverride = 12 },
	{ map = "sp_a2_laser_over_goo", arrival = "aperture_appear_vert.bik", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_catapult_intro", arrival = "faithplate.bik", departure = "faithplate.bik", typeOverride = 6 },
	{ map = "sp_a2_trust_fling",	arrival = "faithplate.bik", departure = "faithplate.bik", typeOverride = 6 },
	{ map = "sp_a2_pit_flings",	arrival = "aperture_appear_vert.bik", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_fizzler_intro",	arrival = "fizzler.bik", departure = "fizzler.bik", typeOverride = 6 },
	{ map = "sp_a2_sphere_peek",	arrival = "aperture_appear_vert.bik", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_ricochet",	arrival = "aperture_appear_vert.bik", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_bridge_intro",	arrival = "hard_light.bik", departure = "hard_light.bik", typeOverride = 12 },
	{ map = "sp_a2_bridge_the_gap", arrival = "hard_light.bik", departure = "hard_light.bik", typeOverride = 6 },
	{ map = "sp_a2_turret_intro",	arrival = "turret_exploded_grey.bik", departure = "", typeOverride = 6 },
	{ map = "sp_a2_laser_relays",	arrival = "", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_turret_blocker",	arrival = "turret_exploded_grey.bik", departure = "turret_exploded_grey.bik", typeOverride = 6 },
	{ map = "sp_a2_laser_vs_turret",arrival = "turret_colours_type.bik", departure = "turret_colours_type.bik", typeOverride = 6 },
	{ map = "sp_a2_pull_the_rug",	arrival = "aperture_appear_vert.bik", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_column_blocker", arrival = "turret_dropin.bik", departure = "turret_dropin.bik", typeOverride = 6 },
	{ map = "sp_a2_laser_chaining", arrival = "turret_colours_type.bik", departure = "turret_colours_type.bik", typeOverride = 6 },
	{ map = "sp_a2_triple_laser",	arrival = "aperture_appear_vert.bik", departure = "aperture_appear_vert.bik", typeOverride = 9 },
	{ map = "sp_a2_bts1",			arrival = "aperture_appear_vert.bik", departure = "", typeOverride = 9 },
	{ map = "sp_a4_intro",			arrival = "", departure = "plc_blue_horiz.bik", typeOverride = 6 },
	{ map = "sp_a4_tb_intro",		arrival = "exercises_horiz.bik", departure = "exercises_horiz.bik", typeOverride = 6 },
	{ map = "sp_a4_tb_trust_drop",	arrival = "plc_blue_horiz.bik", departure = "plc_blue_horiz.bik", typeOverride = 6 },
	{ map = "sp_a4_tb_wall_button",	arrival = "", departure = "" },
	{ map = "sp_a4_tb_polarity",	arrival = "exercises_horiz.bik", departure = "exercises_horiz.bik", typeOverride = 6 },
	{ map = "sp_a4_tb_catch",		arrival = "plc_blue_horiz.bik", departure = "plc_blue_horiz.bik", typeOverride = 6 },
	{ map = "sp_a4_stop_the_box",	arrival = "bluescreen.bik", departure = "bluescreen.bik", typeOverride = 14 },
	{ map = "sp_a4_laser_catapult",	arrival = "bluescreen.bik", departure = "bluescreen.bik", typeOverride = 14 },
	{ map = "sp_a4_laser_platform",	arrival = "bluescreen.bik", departure = "", typeOverride = 14 },
	{ map = "sp_a4_speed_tb_catch",	arrival = "", departure = "bluescreen.bik", typeOverride = 14 },
	{ map = "sp_a4_jump_polarity",	arrival = "bluescreen.bik", departure = "bluescreen.bik", typeOverride = 14 },
	{ map = "sp_a4_finale1",		arrival = "bluescreen.bik", departure = "" },
]

ARRIVAL_VIDEO <- 0
DEPARTURE_VIDEO <- 1
ARRIVAL_DESTRUCTED_VIDEO <- 2
DEPARTURE_DESTRUCTED_VIDEO <- 3

OVERRIDE_VIDEOS <- 0

FIRST_CLEAN_MAP <- "sp_a2_catapult_intro"
randVidIndex <- -1;

function Precache()
{
	local hasPrecache = false;
	if( "PrecachedVideos" in this )
	{
		// don't do anything
	}
	else
	{
		// If we're in a community map, pick a random one
		local communityMapIndex = GetMapIndexInPlayOrder();
		if ( communityMapIndex != -2 )
		{
			if ( communityMapIndex == -1 )
			{
				communityMapIndex = GetNumMapsPlayed()
			}
			
			local movieName = "media\\" + RandomVideos[communityMapIndex % RandomVideos.len()];
			printl( "Preching movie: " + movieName )
			hasPrecache = true;
			PrecacheMovie( movieName )		
		}
		else
		{
			// Commenting this line out because it prevents properly re-precaching movies after loading a save game.
			// The cost is that we end up running this code below about 2x too often, but it's fairly cheap and not realtime code anyways...
			//::PrecachedVideos <- 1
	
			local mapName = GetMapName()
			foreach (index, level in ElevatorVideos)
			{
				if (level.map == mapName)
				{
					local movieName
					if ("additional" in level && level.additional != "" )	
					{
						movieName = "media\\" + level.additional
						//printl( "Preching movie: " + movieName )
						PrecacheMovie( movieName )
						hasPrecache = true;
					}
					
					if ("arrival" in level && level.arrival != "" )	
					{
						movieName = "media\\"
						if( OVERRIDE_VIDEOS == 1 ) 
							movieName += "entry_emergency.bik"
						else
							movieName += level.arrival
					
						//printl( "Preching movie: " + movieName )
						PrecacheMovie( movieName )
						hasPrecache=true;
					}
					
					if ("departure" in level && level.departure != "" )	
					{
						movieName = "media\\"
						if( OVERRIDE_VIDEOS == 1 ) 
							movieName += "exit_emergency.bik"
						else
							movieName += level.departure
					
						//printl( "Preching movie: " + movieName )
						PrecacheMovie( movieName )
						hasPrecache = true;
					}
				}
			}
		}
		
		if(!hasPrecache && communityMapIndex==-2) // Added, are we in an unpublished map and failed to get a video?
		{
			randVidIndex = RandomInt(0,RandomVideos.len()-1);
		}
	}
}

// stubs to supress error - will delete these soon.
function StopEntryVideo(width,height)
{
}

function StopExitVideo(width,height)
{
}

function StartEntryVideo(width,height)
{
}

function StartExitVideo(width,height)
{
}

function StartDestructedEntryVideo(width,height)
{
}

function StartDestructedExitVideo(width,height)
{
}

//============================

function StopArrivalVideo(width,height)
{
	EntFire("@arrival_video_master", "Disable", "", 0)
	EntFire("@arrival_video_master", "killhierarchy", "", 1.0)
	StopVideo(ARRIVAL_VIDEO,width,height)
}

function StopDepartureVideo(width,height)
{
	EntFire("@departure_video_master", "Disable", "", 0)
	EntFire("@departure_video_master", "killhierarchy", "", 1.0)
	StopVideo(DEPARTURE_VIDEO,width,height)
}

function StopVideo(videoType,width,height)
{
	for(local i=0;i<width;i+=1)
	{
		for(local j=0;j<height;j+=1)
		{
			local panelNum = 1 + width*j + i
			local signName
			
			if (videoType == DEPARTURE_VIDEO || videoType == DEPARTURE_DESTRUCTED_VIDEO )
			{
				signName = "@departure_sign_" + panelNum
			}
			else
			{
				signName = "@arrival_sign_" + panelNum
			}
			
			EntFire(signName, "Disable", "", 0)
			EntFire(signName, "killhierarchy", "", 1.0)
		}
	}
}

function StartArrivalVideo(width,height)
{
	StartDestructedArrivalVideo(width,height)
	
//	EntFire("@arrival_video_master", "Enable", "", 0)
//	StartVideo(ENTRANCE_VIDEO,width,height)
}

function StartDepartureVideo(width,height)
{
	StartDestructedDepartureVideo(width,height)

//	EntFire("@departure_video_master", "Enable", "", 0)
//	StartVideo(DEPARTURE_VIDEO,width,height)
}

function StartDestructedArrivalVideo(width,height)
{
	local videoName = ""
	local playDestructed = true

	// If we're in a community map, pick a random one
	local communityMapIndex = GetMapIndexInPlayOrder()
	if ( communityMapIndex != -2 )
	{	
		if ( communityMapIndex == -1 )
		{
			communityMapIndex = GetNumMapsPlayed()
		}
			
		playDestructed = false
		videoName = "media\\" + RandomVideos[communityMapIndex % RandomVideos.len()]
		// reprintl("Setting arrival movie to " + videoName )
	}
	else
	{
		local mapName = GetMapName()
		
		foreach (index, level in ElevatorVideos)
		{
			if (FIRST_CLEAN_MAP == level.map )
			{
				playDestructed = false
			}
			
			if (level.map == mapName && ("arrival" in level) )
			{
				if( level.arrival == "" )
					return
					
				videoName = "media\\"
				
				if( OVERRIDE_VIDEOS == 1 ) 
					videoName += "entry_emergency.bik"
				else
					videoName += level.arrival					
				
				break
			}
		}
	}	
	if (videoName == "" && randVidIndex !=0) // Added, do we have an overriden video to play instead?
	{
	printl("DLC2 SCRIPT OVERRIDE - CHOOSING RANDOM ARRIVAL VIDEO!!");
	videoName="media\\" + RandomVideos[randVidIndex];
	}
	// If we have something to play, do so
	if ( videoName != "" )
	{
		printl("Setting arrival movie to " + videoName )
		EntFire("@arrival_video_master", "SetMovie", videoName, 0)
	
		EntFire("@arrival_video_master", "Enable", "", 0.1)
		StartVideo(playDestructed ? ARRIVAL_DESTRUCTED_VIDEO : ARRIVAL_VIDEO, width, height)
	}
}

function StartDestructedDepartureVideo(width,height)
{
	local playDestructed = true
	local videoName = "";

	// If we're in a community map, pick a random one
	local communityMapIndex = GetMapIndexInPlayOrder()
	if ( communityMapIndex != -2 )
	{	
		if ( communityMapIndex == -1 )
		{
			communityMapIndex = GetNumMapsPlayed()
		}
			
		playDestructed = false;
		videoName = "media\\" + RandomVideos[communityMapIndex % RandomVideos.len()]
		// reprintl("Setting arrival movie to " + videoName )
	}
	else
	{
		local mapName = GetMapName()
		foreach (index, level in ElevatorVideos)
		{
			if (FIRST_CLEAN_MAP == level.map )
			{
				playDestructed = false
			}
			
			if (level.map == mapName && ("departure" in level) )
			{
				if( level.departure == "" )
					return
	
				videoName = "media\\"
				if( OVERRIDE_VIDEOS == 1 ) 
					videoName += "exit_emergency.bik"
				else
					videoName += level.departure
					
				break
			}
		}
	}
	if (videoName == "" && randVidIndex !=0) // Added, do we have an overriden video to play instead?
	{
	printl("DLC2 SCRIPT OVERRIDE - CHOOSING RANDOM DEPARTURE VIDEO!!");
	videoName="media\\" + RandomVideos[randVidIndex];
	}	
	if ( videoName != "" )
	{
		//printl("Setting departure movie to " + videoName )
		EntFire("@departure_video_master", "SetMovie", videoName, 0)
		
		EntFire("@departure_video_master", "Enable", "", 0.1)
		StartVideo(playDestructed ? DEPARTURE_DESTRUCTED_VIDEO : DEPARTURE_VIDEO, width, height)
	}
}

function StartVideo(videoType,width,height)
{
	local videoScaleType = 0
	local randomDestructChance = 0
	
	if( videoType == ARRIVAL_DESTRUCTED_VIDEO || videoType == DEPARTURE_DESTRUCTED_VIDEO )
	{
		videoScaleType = RandomInt(1,5)
	}
	else
	{
		videoScaleType = RandomInt(6,7)
	}
		
	local mapName = GetMapName()
	foreach (index, level in ElevatorVideos)
	{
		if (level.map == mapName)
		{
			if ("typeOverride" in level)
			{
				videoScaleType = level.typeOverride
			}
			
			if ("destructChance" in level)
			{
				randomDestructChance = level.destructChance
			}
		}
	}
	
	for(local i=0;i<width;i+=1)
	{
		for(local j=0;j<height;j+=1)
		{
			local panelNum = 1 + width*j + i
			local signName
			
			if (videoType == DEPARTURE_VIDEO || videoType == DEPARTURE_DESTRUCTED_VIDEO )
			{
				signName = "@departure_sign_" + panelNum
			}
			else
			{
				signName = "@arrival_sign_" + panelNum
			}		
					
			{
				if( randomDestructChance > RandomInt(0,100) )
				{
					EntFire(signName, "Kill", "", 0)
					continue
				}
				
				EntFire(signName, "SetUseCustomUVs", 1, 0)
				
				local uMin = (i+0.0001)/(width)
				local uMax = (i+1.0001)/(width)
				local vMin = (j+0.0001)/(height)
				local vMax = (j+1.0001)/(height)
				
				if( videoScaleType == 0 /*full elevator*/ ) 				
				{
				
				}				
				else if( videoScaleType == 1 /*stretch*/ ) 
				{
					uMin = 1.0 - (1.0-uMin)*(1.0-uMin)*(1.0-uMin)
					uMax = 1.0 - (1.0-uMax)*(1.0-uMax)*(1.0-uMax)
				}				

				else if( videoScaleType == 2 /*Mirror*/ ) 
				{					
					uMin = 4*(1.0-uMin)*uMin
					uMax = 4*(1.0-uMax)*uMax					
				}				
				
				else if( videoScaleType == 3 /*Ouroboros*/ )
				{
					uMin = ((i%12)+0.0001)/12
					uMax = ((i%12)+1.0001)/12

					if( ((i)%2) == 1 )
					{
						local temp = uMin
						uMin = uMax
						uMax = temp
					}
				}
				
				else if( videoScaleType == 4 /*Upside down*/ )
				{
					vMin = 0.99999
					vMax = 0.00001
					
					uMin = ((i%3)+0.0001)/3
					uMax = ((i%3)+1.0001)/3					
				}
				
				else if( videoScaleType == 5 /*Tiled*/ )
				{
					vMin = 0.00001
					vMax = 0.99999
					
					uMin = ((i%3)+0.0001)/3
					uMax = ((i%3)+1.0001)/3
				}

				else if( videoScaleType == 6 /*Tiled Really Big*/ )
				{
					uMin = ((i%8)+0.0001)/8
					uMax = ((i%8)+1.0001)/8
				}

				else if( videoScaleType == 7 /*Tiled Big*/ )
				{
					uMin = ((i%12)+0.0001)/12
					uMax = ((i%12)+1.0001)/12
				}

				else if( videoScaleType == 8 /*Tiled Single*/ )
				{
					uMin = 0.0001
					uMax = 0.9999
					vMin = 0.0001
					vMax = 0.9999
				}

				else if( videoScaleType == 9 /*Tiled Double*/ )
				{
					uMin = ((i%2)+0.0001)/2
					uMax = ((i%2)+1.0001)/2
				}

				else if( videoScaleType == 10 /*Two by two*/ )
				{
					vMin = 0.00001
					vMax = 0.99999
					
					uMin = ((i%2)+0.0001)/2
					uMax = ((i%2)+1.0001)/2
				}

				else if( videoScaleType == 11 /*Tiled off 1*/ )
				{
					vMin = 0.00001
					vMax = 0.99999
					
					uMin = (((i+1)%3)+0.0001)/3
					uMax = (((i+1)%3)+1.0001)/3
				}

				else if( videoScaleType == 12 /*Tiled 2x4*/ )
				{
					uMin = ((i%6)+0.0001)/6
					uMax = ((i%6)+1.0001)/6
				}

				else if( videoScaleType == 13 /*Tiled Double - with two blank*/ )
				{
					if( ((i)%4) < 2 )
					{
						uMin = ((i%2)+0.0001)/2
						uMax = ((i%2)+1.0001)/2
					}
					else
					{
						uMin = 0.97
						uMax = 0.97
					}
				}

				else if( videoScaleType == 14 /*bluescreen*/ )
				{
					if( (i%8) >= 1 &&  
						(i%8) < 7 )
					{
						uMin = (((i-1)%8)+0.0001)/6
						uMax = (((i-1)%8)+1.0001)/6
					}
					else
					{
						uMin = 0.97
						uMax = 0.97
					}
				}
								 
				EntFire(signName, "SetUMin", uMin, 0)
				EntFire(signName, "SetUMax", uMax, 0)
				EntFire(signName, "SetVMin", vMin, 0)
				EntFire(signName, "SetVMax", vMax, 0)

				EntFire(signName, "Enable", "", 0.1)
				
//				printl(signName + " " + uMin + " " + uMax + " " + vMin + " " + vMax )
			}
		}
	}
}
