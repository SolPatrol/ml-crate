### Game Overview

- Title: TBD
- **Genre:** CRPG
- **Setting:** *Deadlands: Horror at Headstone Hill*
- **Core System:** SWADE
- Game Engine: BEVY

### Core Design Philosophy

- Solving the Branching Problem
    - Problem
        - When building a CRPG, the more choices you give a player, you increase the amount of content that needs to be created. Due to this, CRPGs usually try bringing players back to the main chain of events or “Golden Path,” in order to limit content to a manageable amount.
    - Our Approach
        - Using Constructive AI, the Agent Orchestra reads the player's entire history and provides infinite cohesive branches on the fly that are all pre-validated assets. We never have to limit content due to resources, and players can do things that the creators would never expect with infinite possibilities
            - Agent evaluates the player’s history vector (choices, flags, relationships, locale, session context).
            - The system assembles new scenes, encounters, and dialogue from a **pre-validated asset library** (locations, NPC kits, stat blocks, dialogue beats, encounter templates, props, VFX/SFX cues).
            - Branches are not limited but **constructed** and gated by rulesets (SWADE mechanics, setting canon, safety/content checks).
- Player Experience
    - The goal is to make everything feel magical. For the player the game will feel boundless—as if every action summons a Game Master who instantly writes new content. All of this will be done with a library of verified assets, orchestrated seamlessly.

### Gameplay Mechanics

- Exploration
    - The map will be a collection of Nodes (locations) and Paths (routes between them). The Agent will decide what the player sees and what state the nodes/paths are in.
    - The "Mapmaker" Agent would constantly check and/or control these factors:
        - **Player Knowledge (The "Fog of War"):** This is the "live construction" part. Nodes don't appear until the player learns about them.
            - **Rumor:** A "rumor" asset (e.g., "Heard old man Withers has a shack up in the hills") adds a node to the map: **"Withers' Shack (Rumored)"** with a `?` icon.
            - **Clue:** A "clue" asset (e.g., finding Withers' deed) changes the node: **"Withers' Shack."**
        - **Fear Level (The "Supernatural Filter"):** As the Fear Level of a region rises, the AI *assembles* new, scarier assets onto the map.
            - **Low Fear:** The node for "Town Cemetery" looks normal. The path to it is clear.
            - **High Fear:** The *icon* for "Town Cemetery" is replaced with a pre-gen asset tagged `[Supernatural_Glow]`. A *new* node, **"The Hanging Tree,"** (a `[Savage_Tale]` asset) might be assembled *onto* the map, as the barrier between worlds gets thinner.
        - **Player State (The "Personalized" Map):**
            - **Hindrances:** If you have `[Hindrance: Wanted]`, the Agent might assemble a **"Bounty Hunter Camp"** node on the map.
            - **Active Quest:** If you need to see the Assayer, the **"Assayer's Office"** node might pulse with a gentle glow, assembled by the "helpful GM" agent.
        - T**ravel & Encounters:** Moving the "chip" along a Path must be the trigger for encounters.
            - **Cost of Travel:** Moving between nodes could cost **Time** (potential resource). Moving from Heaston Hill to the Mines might take 4 hours.
            - **The "Interruption":** When travel is initiated, the Agent runs a check.
            `[Path: Forest]` + `[Time: Night]` + `[Fear_Level: High]` = High probability of retrieving a `[Encounter: Undead]` asset.
            - **Loading the Battlemap:** When an encounter is assembled, the game loads a pre-gen 2D battlemap tagged appropriately.
        - **Dynamic Node States:** Nodes shouldn't just be *on* or *off*. They need states that the AI controls.
            - The **"Heaston Hill"** node could have states like:
                - `[State: Normal]`
                - `[State: Boomtown_Fever]` (More people, higher prices)
                - `[State: Riot]` (Triggered by story events, node is locked or dangerous)
                - `[State: Haunted]` (After a major supernatural event)
            - The AI's job is to update these states on the map, which in turn changes what assets (NPCs, quests) are available when you "enter" the node.
        - **Environmental Effects:** The agent should assemble environmental hazards onto the Paths.
            - A major story beat could cause the Agent to retrieve a **"Rockslide"** asset, place it on the Path to the mine, and tag that path as `[Blocked]`.
            - The Agent can assemble weather. A `[Blizzard]` asset could cover the whole map, doubling all travel times and increasing the chance of `[Encounter: Freezing]` (a survival-based encounter).
- Combat
    - Combat is initiated when the player's "chip" on the Living Atlas map collides with an encounter (either player-initiated or assembled by the AI as an "interruption").
        - **Process:** The game "zooms in" from the atlas view to a 2D Tactical Battlemap.
        - **Battlemap Assembly:** The **Agent Orchestra** assembles the battlefield.
            - **Retrieves Map:** It retrieves a pre-made 2D battlemap asset tagged by the node's context (e.g., `[Forest_Path]`, `[Mine_Tunnel]`, `[Saloon_Interior]`).
            - **Retrieves Actors:** It retrieves the `[Enemy: Stat_Block]` assets for the encounter.
            - **Retrieves Scenery (Cover):** It intelligently retrieves and places relevant `[Cover: Asset]` (e.g., `[Crates]`, `[Fallen_Logs]`, `[Overturned_Table]`). This ensures the "constructed" battlemap is tactically unique every time.
    - Action Deck
        - At the start of each round, the UI visually deals Action Cards from a standard 52-card deck to the player and all major enemies/allies.
            - Drawing Jokers is a major event:
                - Still need to determine how this works but could be a standard buff: i.e., Player gets +2 to all Trait rolls and damage this round.
                - Also need to determine how the Agent can make each Joker pull unique (but this might too much)
    - Actions
        - From a top-down view, the player's chip has a **Pace** (movement distance, shown as a circle). They can move and perform one action.
        - **Key Actions:**
            - **Shoot/Fight:** Make a Trait roll (Shooting/Fighting) vs. enemy's Parry/Toughness.
            - **Test:** Agents can query Knowledge Base and assemble unique, pre-written lines that the moment. Unique responses for Opposed Roll.
                - *Player:* Selects "Test (Taunt)" on an enemy.
                - *Agent Orchestra:* Retrieves a relevant `[Taunt: Line]` for the player chip to "say."
                - *If Successful:* The enemy chip gets a `[Status: Distracted/Vulnerable]` tag, which their own AI *must* obey (e.g., they can't use actions, or all attacks against them are +2).
            - **Support/Hinder:** Give an ally a bonus or an enemy a penalty.
        - **Gridless Movement:** Movement is freeform within the Pace radius, not locked to a grid.
        - **Multi-Actions:** The player can attempt multiple actions (e.g., shoot twice) by taking a stacking -2 penalty.
    - Damage and Bennies
        - **Shaken:** Most successful attacks make a target "Shaken."
            - *Mechanic:* A Shaken chip can *only* use its turn to attempt a Spirit roll to "un-Shake." They cannot move or act. This makes combat swingy and fast.
            - *UI:* The chip visually "rattles" or is grayed out.
        - **Wounds:** If you hit a target that is *already* Shaken, you cause a Wound.
            - *Mechanic:* Player/NPCs have 3 Wounds. Taking a Wound causes a penalty to all rolls. At 4 Wounds, you're Incapacitated.
            - *UI:* A 3-pip "Wound Tracker" on the chip.
        - **Bennies:**
            - *Reroll:* After any failed Trait roll, the player can spend a Benny to reroll.
            - *Soak:* When the player takes a Wound, they can spend a Benny to make a Vigor roll to "Soak" it (negate it).
            - The Agent can award Bennies for in-combat roleplaying. If a player with `[Hindrance: Heroic]` uses an action to "Support" an ally instead of attacking, the GM Agent can award a Benny with the note "Heroic Act!"
    - Additional Agent Roles
        - Enemy behavior is 90% rules-based. It follows standard (non-generative) CRPG logic to be a fair, predictable challenge. (e.g., *If Shaken, try to un-Shake. If player is Vulnerable, attack them.*)
        - Combat Director via Constructive AI
            - Agent watches the *results* of the deterministic rolls and assembles cinematic text.
                - *Player Rolls a Raise (Critical Hit):* AI retrieves `[Desc: Gory_Hit]` + `[Weapon: Revolver]` -> Assembles: "Your shot punches clean through the outlaw, sending a spray of red mist into the dust."
                - *Player Fails (Critical Miss):* AI retrieves `[Desc: Misfire]` + `[Weapon: Revolver]` -> Assembles: "Click. Your hammer falls on a dead primer. The cultist grins."
                - *Dynamic Terrain:* If a spell `[Spell: Blast(Fire)]` is used, the AI can retrieve and place a `[Terrain: On_Fire]` asset, creating a new, dynamic hazard on the battlemap.

### Narrative Structure

- Clue Scroll / Web
    - A UI screen that visually maps the player's understanding of the mystery. It's a graph of interconnected nodes (Clues, People, Locations).
    - **How it's Populated:**
        - When the player "finds" a clue (by passing an Investigation check, winning a combat, or through dialogue), that clue is added to their Web.
        - **Example:** Finding a *Strange Black Nugget* on a dead miner adds the **`[Clue: Strange_Nugget]`** node to the Web.
    - Clues are dynamic with the enviroment:
        - A Clue is *not* just text. It is a **"Topic Key"** that is permanently added to the player's "Stance & Topic" dialogue options.
        - Once you have the **`[Clue: Strange_Nugget]`** key, you can go to *any* NPC in the game and use it as a Topic (e.g., "Ask about Strange Nugget").
        - The Agent Orchestra then assembles a unique response based on *that NPC's* profile (e.g., The Assayer is curious, the Sheriff is suspicious, the hidden Villain is evasive).
- The *Headstone Hill* book is already structured as a main plot surrounded by "Savage Tales" (modular side quests).
    - **Deconstruction:** Each Savage Tale from the book (e.g., "The Ghost of Jebediah") is deconstructed and made into a "Narrative Asset" containing:
        - **Triggers:** The conditions the AI needs to meet to "assemble" this quest (e.g., `[Time: Night]`, `[Location: Cemetery]`, `[Fear_Level: 3+]`).
        - **Actors:** The enemy/NPC stat blocks involved.
        - **Clues:** The "Topic Keys" the player gets as a reward.
        - **Maps:** The 2D battlemap(s) to be loaded.
    - **Assembly:** The Agent is constantly scanning the player's state. When the player's chip moves to the Cemetery at night, the AI sees the triggers are met, retrieves the "Ghost of Jebediah" asset, and "constructs" the encounter on the Living Atlas map.
- Every major NPC has a **Profile** (a JSON file) that the Agent Orchestra reads in real-time.
    - **Example Profile Schema:**
        - `"Name": "Sheriff 'Hoss' Brody"`
        - `"Disposition_Tags": ["Lawful", "Suspicious", "Brave", "Weary"]`
        - `"Knowledge_Topics": ["Missing_Miners", "Town_Politics", "Bar_Fight_Last_Night"]`
        - `"Secret_Knowledge": ["Fears_The_Mine", "Suspects_The_Mayor"]`
    - **How it Works (Dialogue):**
        - **Player:** Selects `[Stance: Sympathetic]` + `[Topic: Missing_Miners]`
        - **Agent:** Reads the Player's vector and the Sheriff's Profile.
        - **Query:** "Retrieve asset tagged: `[Dialogue: Sheriff]` + `[Sympathetic]` + `[Topic: Missing_Miners]` + `[Disposition: Weary]`"
        - **Assembly:** The Agent retrieves a pre-written, voice-acted line like, *"It's a bad business, son. We've lost six men... and I ain't got a single lead to give their families."*
        - **Unlocking Secrets:** If the player gets a "Raise" on a Persuasion roll, the Agent is permitted to retrieve a line from the `"Secret_Knowledge"` pool.

### Character / Progression System

- Character Creation
    - Players create “Wild Cards” using normal SWADE process
        - **Archetypes (Quick-Start):** Players can select from pre-built Deadlands archetypes to get started quickly:
            - **Gunslinger:** Combat focus (Shooting, Agility).
            - **Huckster:** Arcane caster (Spellcasting, Smarts, gambles with demons).
            - **Blessed:** Divine caster (Faith, Spirit, channels miracles).
            - **Mad Scientist:** Tech caster (Weird Science, Smarts, builds gadgets).
        - **Attributes (The Core):** Players assign points to the 5 core SWADE attributes. These start at a `d4` die type.
            - **Agility:** Physical finesse, Shooting, Fighting.
            - **Smarts:** Mental acuity, Investigation, Spellcasting.
            - **Spirit:** Inner will, Faith, Fear checks, recovering from Shaken.
            - **Strength:** Raw power, melee damage, physical feats.
            - **Vigor:** Health, resisting poison/disease, soaking Wounds.
        - **Skills (The "Know-How"):** Players spend points to buy skills (e.g., *Shooting, Persuasion, Investigation, Notice*). A skill cannot be higher than its linked attribute.
        - **Edges & Hindrances (The "AI Hooks"):** This is the most critical step for our AI.
            - **Hindrances (Flaws):** Players *must* select Hindrances (e.g., *Greedy, Wanted, Phobia, Loyal*). These are the AI's primary "interrupt" triggers.
            - **Edges (Perks):** Players select positive perks (e.g., *Ace, Quick Draw, Arcane Background*).
- Character Sheets In-Game
    - Agent reads the character sheet as a set of semantic "tags" to filter content assembly.
        - **Attributes as Competence:** The die type (d4-d12) acts as a "competency filter" for non-roll actions.
            - *Example:* Player with `[Smarts: d4]` examines a complex machine -> Agent retrieves text: "It's a mess of gears and wires... beyond you."
            - *Example:* Player with `[Smarts: d12]` examines the same machine -> Agent retrieves text: "You see it clearly... the pressure valve has been deliberately sabotaged."
        - **Skills as "Topic Keys":** Having a skill unlocks passive information and active dialogue options.
            - *Example:* Player with `[Skill: Investigation >= d8]` enters a crime scene -> Agent *automatically* assembles and reveals clues (e.g., "The powder burns smell of brimstone...") that a d4 character would never see.
        - **Hindrances as "Encounter Subscriptions":**
            - `[Hindrance: Wanted]` -> The agent is authorized to retrieve and assemble `[Encounter: Bounty_Hunter]` on the Living Atlas map.
            - `[Hindrance: Greedy]` -> The Agent assembles new, risk/reward dialogue options tagged `[Greedy]` that other characters don't get.
- Progression
    - **Earning Advances:** The player earns an "Advance" point after completing a major "Savage Tale" (side quest), a key beat in the *Headstone Hill* mystery, or (TBD) after a set number of in-game days.
    - **Spending Advances:** An Advance point can be spent to do **one** of the following:
        1. Increase a Skill by one die type.
        2. Buy a new Edge (if prerequisites are met).
        3. Increase an Attribute by one die type (this is more "expensive" and only allowed at set milestones, e.g., every 4 Advances).
    - **Evolving AI State:** As the player's sheet evolves, the Agents content filter evolves with it.
        - *Player buys the `[Edge: Danger_Sense]`.*
        - *AI Translation:* The "Storyteller" agent now gains permission to assemble pre-gen `[Flavor_Text: Warning]` assets (e.g., "The hair on your neck stands up...") *before* an ambush encounter is triggered.

### Art Direction and Assets

- Visual style: TBD
    - The players should *feel* the world change. For example the art will be the primary mechanism for “Fear Level”
- The 2D map is the main game board and must be a dynamic, "living" asset.
    - **The Base Map:** A high-resolution, hand-drawn topographical map of the *Headstone Hill* region. It looks like a beautiful, static piece of art.
    - **Nodes (Locations):** These are not just dots. They are small, hand-drawn "vignette" icons that represent the location (e.g., a "Saloon" icon, a "Mine Entrance" icon).
    - **Player "Chip":** A top-down "pawn" or token, customized with the player's portrait.
    - **Dynamic Assets Example:** We do not have *one* icon for the Saloon. We have an "Asset Pack" for it:
        - `[Saloon_Icon_Normal]`
        - `[Saloon_Icon_Riot]` (e.g., smoke, broken windows)
        - `[Saloon_Icon_Haunted]` (e.g., glowing with green mist)
        - `[Saloon_Icon_Closed]` (e.g., boarded up)
        - When the Agent changes the state of the Saloon node, it *retrieves and assembles* the correct icon onto the map, visually informing the player that the world has changed.
- Battlemaps
    - When combat is initiated, the game "zooms in" to a pre-built 2D tactical map.
        - **Style:** Clean, 2D top-down, gridless (using movement circles). Readability is the top priority.
        - **Map Library:** The "Knowledge Base" contains a library of pre-made battlemaps, each tagged by context.
            - `[Map: Saloon_Interior]`, `[Map: Forest_Clearing]`, `[Map: Mine_Tunnel]`
        - The Agent retrieves the correct base map and then "populates" it by retrieving dynamic `[Cover_Asset]` (e.g., `[Crate]`, `[Overturned_Table]`, `[Fallen_Log]`) to make each fight on the same map feel unique.
- Characters
    - **Portraits:** High-detail, hand-drawn (style) portraits for all major NPCs and the player. This is where the emotion of the dialogue is conveyed. Portaits will be dynamic and updated based on conditional states.
    - **Tactical "Chips":** In combat, player and enemy "chips" must clearly display their current state (i.e., a "rattled" visual for **Shaken**, a red-tinted border for **Wounded**).
    - **The "Tabletop" HUD:** The UI must be clean and tactile, reinforcing the tabletop feel.
        - **Bennies:** A visible, physical-looking pile of poker chips in the corner.
        - **Action Cards:** The Initiative "deal" at the start of combat must be a major, satisfying visual event.
        - **Wound Tracker:** A 3-pip "Wound" tracker (e.g., 3 bullet holes) instead of a health bar.
- Fear Level
    - This is the most critical art requirement. The Agents "Fear Level" (0-6) acts as a global filter for asset retrieval.
        - Assets in *at least three* states: `[Fear_0-2: Normal]`, `[Fear_3-4: Spooky]`, and `[Fear_5-6: Corrupted]`
        - **Examples:**
            - **Map Filter:** At Fear 5+, the AI retrieves a `[Map_Filter: Sickly_Green]` to overlay the entire Living Atlas.
            - **Node Icons:** The "Town" icon changes from `[Town_Normal]` to `[Town_Corrupted]` (e.g., icons of gallows, shadowy figures in alleys).
            - **Music:** The audio engine (part of the Orchestra) retrieves `[Music: Normal_Town]` or `[Music: Haunted_Town]`.
            - **Battlemaps:** The AI retrieves `[Scenery: Normal_Tree]` or `[Scenery: Bleeding_Tree]` to populate the battlemap.
    - Agent can dynamically “art direct” the entire game based on the narrative state, making the world look and feel more haunted as the players uncover mysteries