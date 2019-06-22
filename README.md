# HammerAddons

Various improvements for Portal 2's Hammer. To install, [download][dl_zip] the repro and merge with your `Portal 2/` folder.

* FGD Updates:
	* Lots and lots of changes, with [Skotty's FGDs][skotty] as a base.
	* Adds new models for  `prop_indicator_panel`, `prop_portal` and `info_placement_helper`.
	* Adds sprites for many models, from a number of sources:
		* [The TF2 Ultimate Mapping Resource Pack][tf2]
		* [ZPS: Supplemental Hammer Icons][zps]
		* [ts2do's HL FGDs][ts2do]
	* Add inputs and keyvalues supported by entities but not in the default list
	* Adds several 'fake' keyvalues to allow cubes, funnels and turrets to display with their correct models.
	* Adds lots more AutoVisgroups for easily hiding entities.
* Tweak `video_splitter.nut` to display random videos on unpublished maps.
* Tweak `dlc2_vo.nut` to preview random Cave lines in unpublished maps if set to do so in  `global_pti_ents`.
* Notify more visibly when the vote screen is triggered.

[dl_zip]: https://github.com/TeamSpen210/HammerAddons/archive/master.zip
[skotty]: http://forums.thinking.withportals.com/downloads.php?view=detail&df_id=507
[tf2]: http://forums.tf2maps.net/showthread.php?t=4674
[ts2do]: http://halflife2.filefront.com/file/HalfLife_2_Upgraded_Base_FGDs;48139
[zps]: http://www.necrotalesgames.com/tools/index.php
