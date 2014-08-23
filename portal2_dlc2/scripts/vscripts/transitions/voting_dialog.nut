function callvote()
{
	RequestMapRating()
	printl("Voting for: '" + GetMapName() + "'" )
	if (GetMapIndexInPlayOrder()==-2) // -2 = not on workshop, -1=not played before, 0+ = index of map.
		{
		ScriptShowHudMessageAll("|- Insert Vote Screen Here -|",3)
		}
}