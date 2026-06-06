//! Ingenious Hunter standby / control-panel window.
//!
//! Shown whenever no League game is running.  Renders a full-screen neon-red /
//! steel / black control panel where the user can browse champions and apply
//! rune pages to the live League Client via the LCU API.
//!
//! # Architecture
//!
//! [`ControlPanelState`] is shared (via `Arc<Mutex<>>`) between:
//! - the UI thread (`render_control_panel_into` called from `on_pre_paint`)
//! - the rune worker thread (mutates `rune_status` after each LCU write)
//!
//! Click handlers send [`RuneCmd`]s over a sync channel; the worker owns the
//! receiving end.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use brick::{BrickEvent, BrickRenderer, NativeNode, NativeRenderer, WindowCmd};
use cathedral_rift::patch_notes::{classify_line, LineKind};
use cathedral_rift::{BuildRecommendation, LiveRecommender, PatchChange, RuneRecommendation};

use crate::champ_select_auto::AutomationConfig;
use crate::theme::Theme;

// ── Build tag — bump this each patch so the running build is visually identifiable ──

pub const BUILD: &str = "B10";

// ── Window dimensions ────────────────────────────────────────────────────────

pub const PANEL_W: u32 = 1440;
pub const PANEL_H: u32 = 900;

/// Custom title-bar height (replaces the OS chrome — decorations: false).
const TITLEBAR_H: u32 = 64;
const SIDEBAR_W:  u32 = 180;

// ── Champion roster (matches assets/champion_icons/*.png, Ruby_ skins excluded) ──

static ALL_CHAMPIONS: &[&str] = &[
    "Aatrox","Ahri","Akali","Akshan","Alistar","Ambessa","Amumu","Anivia","Annie","Aphelios",
    "Ashe","AurelionSol","Aurora","Azir","Bard","Belveth","Blitzcrank","Brand","Braum","Briar",
    "Caitlyn","Camille","Cassiopeia","Chogath","Corki","Darius","Diana","DrMundo","Draven","Ekko",
    "Elise","Evelynn","Ezreal","FiddleSticks","Fiora","Fizz","Galio","Gangplank","Garen","Gnar",
    "Gragas","Graves","Gwen","Hecarim","Heimerdinger","Hwei","Illaoi","Irelia","Ivern","Janna",
    "JarvanIV","Jax","Jayce","Jhin","Jinx","KSante","Kaisa","Kalista","Karma","Karthus",
    "Kassadin","Katarina","Kayle","Kayn","Kennen","Khazix","Kindred","Kled","KogMaw","Leblanc",
    "LeeSin","Leona","Lillia","Lissandra","Lucian","Lulu","Lux","Malphite","Malzahar","Maokai",
    "MasterYi","Mel","Milio","MissFortune","MonkeyKing","Mordekaiser","Morgana","Naafiri","Nami",
    "Nasus","Nautilus","Neeko","Nidalee","Nilah","Nocturne","Nunu","Olaf","Orianna","Ornn",
    "Pantheon","Poppy","Pyke","Qiyana","Quinn","Rakan","Rammus","RekSai","Rell","Renata",
    "Renekton","Rengar","Riven","Rumble","Ryze","Samira","Sejuani","Senna","Seraphine","Sett",
    "Shaco","Shen","Shyvana","Singed","Sion","Sivir","Skarner","Smolder","Sona","Soraka",
    "Swain","Sylas","Syndra","TahmKench","Taliyah","Talon","Taric","Teemo","Thresh","Tristana",
    "Trundle","Tryndamere","TwistedFate","Twitch","Udyr","Urgot","Varus","Vayne","Veigar",
    "Velkoz","Vex","Vi","Viego","Viktor","Vladimir","Volibear","Warwick","Xayah","Xerath",
    "XinZhao","Yasuo","Yone","Yorick","Yunara","Yuumi","Zaahen","Zac","Zed","Zeri","Ziggs",
    "Zilean","Zoe","Zyra",
];

/// Pre-warm the brick image cache for all champion icons at the four sizes
/// used in the control panel UI.  Spawns a background thread; safe to call
/// before the render loop starts.
pub fn warm_image_cache() {
    // Sizes used: rune-tab header (200×200), picker grid (140×140),
    // champion strip portrait (120×120), automation roster picker (80×80).
    const SIZES: &[(u32, u32)] = &[(200, 200), (140, 140), (120, 120), (80, 80)];
    let entries: Vec<(String, u32, u32)> = ALL_CHAMPIONS
        .iter()
        .flat_map(|&name| {
            SIZES.iter().map(move |&(w, h)| {
                (format!("assets/champion_icons/{name}.png"), w, h)
            })
        })
        .chain([
            ("assets/IH_Icon_simple.png".to_string(), 40, 40),
        ])
        .collect();
    NativeRenderer::preload_images(entries);
}

/// Keystones, secondary runes, and row-level rune entries that appear as h3
/// headers in Riot patch notes.  Used to distinguish rune changes from item
/// changes in the global patch overview.
static RUNE_NAMES: &[&str] = &[
    // Precision keystones + rows
    "Conqueror","Lethal Tempo","Fleet Footwork","Press the Attack",
    "Triumph","Legend: Alacrity","Legend: Haste","Legend: Bloodline",
    "Coup de Grace","Cut Down","Last Stand",
    // Domination
    "Electrocute","Dark Harvest","Hail of Blades","Predator",
    "Cheap Shot","Taste of Blood","Sudden Impact",
    "Zombie Ward","Ghost Poro","Eyeball Collection",
    "Treasure Hunter","Ingenious Hunter","Relentless Hunter","Ultimate Hunter",
    // Sorcery
    "Arcane Comet","Phase Rush","Summon Aery",
    "Nullifying Orb","Manaflow Band","Nimbus Cloak",
    "Transcendence","Celerity","Absolute Focus","Scorch","Waterwalking","Gathering Storm",
    // Inspiration
    "First Strike","Glacial Augment","Unsealed Spellbook",
    "Magical Footwear","Perfect Timing","Biscuit Delivery","Cosmic Insight",
    "Approach Velocity","Time Warp Tonic",
    // Resolve
    "Grasp of the Undying","Aftershock","Guardian",
    "Demolish","Font of Life","Shield Bash","Conditioning","Second Wind",
    "Bone Plating","Overgrowth","Revitalize","Unflinching",
    // Category-level headers that sometimes appear as h3
    "Runes","Rune Changes",
];

/// Reduce a name to lowercase alphanumeric only (same logic as patch_notes::slug).
fn name_slug(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Patch notes use display names; Data Dragon uses internal keys for some champs.
fn ddragon_key(display: &str) -> &str {
    match name_slug(display).as_str() {
        "wukong"       => "MonkeyKing",
        "nunuwillump"  => "Nunu",
        "renataglascc" => "Renata",
        _              => display,
    }
}

fn is_champion_entry(name: &str) -> bool {
    let target = name_slug(ddragon_key(name));
    ALL_CHAMPIONS.iter().any(|&c| name_slug(c) == target)
}

fn is_rune_entry(name: &str) -> bool {
    let target = name_slug(name);
    RUNE_NAMES.iter().any(|&r| name_slug(r) == target)
}

#[derive(PartialEq)]
enum PatchClass { Buff, Nerf, Neutral }

fn patch_change_class(summary: &str) -> PatchClass {
    let s = summary.to_ascii_lowercase();
    let buffs  = ["increased", "buffed", "improved", "bonus", "added", "stronger", "gains", "enhanced"]
        .iter().filter(|&&k| s.contains(k)).count();
    let nerfs  = ["reduced", "decreased", "nerfed", "lowered", "removed", "shorter", "weaker", "lost"]
        .iter().filter(|&&k| s.contains(k)).count();
    if buffs > nerfs { PatchClass::Buff }
    else if nerfs > buffs { PatchClass::Nerf }
    else { PatchClass::Neutral }
}

static SYSTEM_TERMS: &[&str] = &[
    "Baron", "Dragon", "Rift Herald", "Void Grub", "Jungle",
    "Minion", "Turret", "Nexus", "Inhibitor", "Scuttle", "Tower",
    "Elemental", "Infernal", "Mountain", "Ocean", "Cloud", "Hextech",
    "Chemtech", "Atakhan", "Map", "Vision", "Gold",
];

fn is_patch_system_entry(name: &str) -> bool {
    SYSTEM_TERMS.iter().any(|&t| name.to_ascii_lowercase().contains(&t.to_ascii_lowercase()))
}

fn item_icon_id(name: &str) -> Option<u32> {
    Some(match name_slug(name).as_str() {
        // Starters
        "doransshield"          => 1054,
        "doransblade"           => 1055,
        "doransring"            => 1056,
        // Boots
        "berserkersgreaves"     => 3006,
        "bootsofswiftness"      => 3009,
        "sorcerersshoes"        => 3020,
        "platedsteelcaps"       => 3047,
        "mercurystreads"        => 3111,
        "mobilityboots"         => 3117,
        // Completed items — alphabetical
        "abyssalmask"           => 3001,
        "anathemaschain"        => 3330,
        "archangelsstaff"       => 3003,
        "banshesveil"           => 3102,
        "blackcleaver"          => 3071,
        "bloodthirster"         => 3072,
        "cosmicdrive"           => 4629,
        "cryptbloom"            => 6667,
        "deadmansplate"         => 3742,
        "deathsdance"           => 6333,
        "demonicembrace"        => 4637,
        "divinesunderer"        => 4005,
        "edgeofnight"           => 3814,
        "everfrost"             => 6656,
        "fimbulwinter"          => 3119,
        "forceofnature"         => 4401,
        "galeforce"             => 6671,
        "gargoylestoneplate"    => 3193,
        "goredrinker"           => 6630,
        "grailofthe undying"    => 3003,
        "graspoftheundying"     => 3437,
        "heartsteel"            => 6664,
        "horizonfocus"          => 4628,
        "hextechrocketbelt"     => 3152,
        "immortalshieldbow"     => 6673,
        "infinityedge"          => 3031,
        "jakshotheprotean"      => 6665,
        "knightsvow"            => 3109,
        "krakenslayer"          => 6672,
        "liandryanguish"        => 6652,
        "liandrysheart"         => 6652,
        "liandrystorment"       => 6652,
        "locketoftheironsola"   | "locketoftheironsolar" | "locketoftheironsolarilol" | "locketoftheironsolar" => 3190,
        "ludenstempest"         => 6655,
        "malignance"            => 6699,
        "mawofmalmortius"       => 3156,
        "mercurialscimitar"     => 3139,
        "moonstone"             | "moonstonerenewer" => 6617,
        "morellonomicon"        => 3165,
        "nashorstooth"          => 3115,
        "navoriquickblades"     | "navoriflickerblade" => 6692,
        "opportunity"           => 6701,
        "prowlersclaw"          => 6693,
        "rabadonsdeathcap"      => 3089,
        "randuin"               | "randuinsomen" => 3143,
        "ravenoushydra"         => 3074,
        "riftmaker"             => 4633,
        "rodofages"             => 6657,
        "runaanshurricane"      => 3085,
        "rapidfirecannon"       => 3094,
        "rylais"                | "rylaiscrystalscepter" => 3116,
        "serpentsfang"          => 6035,
        "seryldasgrudge"        => 6694,
        "shadowflame"           => 4645,
        "sheen"                 => 3057,
        "shurelyas"             | "shurelyasbattlesong" => 2065,
        "staffofflowingwater"   => 3853,
        "steraksgage"           => 3053,
        "stormsurge"            => 4646,
        "stormrazor"            => 3095,
        "stridebreaker"         => 6631,
        "sunderedsky"           => 6670,
        "sunfireaegis"          => 4630,
        "thecollector"          => 6676,
        "thornmail"             => 3075,
        "titanichydra"          => 3748,
        "trinityforce"          => 3078,
        "turbochemtank"         => 6662,
        "umbralglaive"          => 3179,
        "voidstaff"             => 3135,
        "warmogsarmor"          => 3083,
        "witsend"               => 3091,
        "zhonyashourglass"      => 3157,
        _ => return None,
    })
}

fn perk_icon_id(name: &str) -> Option<u32> {
    let slug = name_slug(name);
    for &id in &[
        8005, 8008, 8009, 8010, 8014, 8017, 8021, 8105, 8106, 8112, 8126,
        8128, 8135, 8137, 8139, 8140, 8141, 8143, 8210, 8214, 8224, 8226,
        8229, 8230, 8232, 8233, 8234, 8236, 8237, 8242, 8275, 8299, 8304,
        8306, 8313, 8316, 8321, 8345, 8347, 8351, 8352, 8360, 8369, 8401,
        8410, 8429, 8437, 8439, 8444, 8446, 8451, 8453, 8463, 8465, 8473,
        9101, 9103, 9104, 9105, 9111, 9923,
        5001, 5002, 5003, 5005, 5007, 5008,
    ] {
        if name_slug(perk_name(id)) == slug { return Some(id); }
    }
    None
}

#[derive(Clone, Copy, Debug)]
enum EntryIconStyle { None, Champion, Item, Rune }

fn patch_overview_row(
    label:   &str,
    total:   usize,
    buffs:   usize,
    nerfs:   usize,
    theme:   &Theme,
    parent:  &NativeNode,
) {
    let row = hrow(40);
    NativeRenderer::append(parent, &row);

    let lbl = NativeRenderer::text(label);
    NativeRenderer::set_attr(&lbl, "data-color",       theme.muted);
    NativeRenderer::set_attr(&lbl, "data-text-size",   "18");
    NativeRenderer::set_attr(&lbl, "data-text-weight", "bold");
    NativeRenderer::set_attr(&lbl, "data-height",      "40");
    NativeRenderer::set_attr(&lbl, "data-w",           "140");
    NativeRenderer::append(&row, &lbl);

    let tot = NativeRenderer::text(&format!("{total} changes"));
    NativeRenderer::set_attr(&tot, "data-color",     theme.text);
    NativeRenderer::set_attr(&tot, "data-text-size", "20");
    NativeRenderer::set_attr(&tot, "data-height",    "40");
    NativeRenderer::set_attr(&tot, "data-w",         "140");
    NativeRenderer::append(&row, &tot);

    let buf_t = NativeRenderer::text(&format!("▲  {buffs} buffs"));
    NativeRenderer::set_attr(&buf_t, "data-color",     "#4caf6e");
    NativeRenderer::set_attr(&buf_t, "data-text-size", "20");
    NativeRenderer::set_attr(&buf_t, "data-height",    "40");
    NativeRenderer::set_attr(&buf_t, "data-w",         "130");
    NativeRenderer::append(&row, &buf_t);

    let nrf_t = NativeRenderer::text(&format!("▼  {nerfs} nerfs"));
    NativeRenderer::set_attr(&nrf_t, "data-color",     "#e05555");
    NativeRenderer::set_attr(&nrf_t, "data-text-size", "20");
    NativeRenderer::set_attr(&nrf_t, "data-height",    "40");
    NativeRenderer::append(&row, &nrf_t);
}

fn patch_entry_row(change: &PatchChange, icon_style: EntryIconStyle, theme: &Theme, parent: &NativeNode) {
    let cls = patch_change_class(&change.summary);
    let badge_color = match cls {
        PatchClass::Buff    => "#4caf6e",
        PatchClass::Nerf    => "#e05555",
        PatchClass::Neutral => "#888899",
    };
    let badge_label = match cls {
        PatchClass::Buff    => "  ▲ BUFF",
        PatchClass::Nerf    => "  ▼ NERF",
        PatchClass::Neutral => "",
    };

    let summary_lines: Vec<&str> = change.summary
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    let row = NativeRenderer::element("div");
    NativeRenderer::set_attr(&row, "data-layout",        "row");
    NativeRenderer::set_attr(&row, "data-fill",          "transparent");
    NativeRenderer::set_attr(&row, "data-border-left",   &format!("{}:3", badge_color));
    NativeRenderer::set_attr(&row, "data-border-bottom", &format!("{}:1", theme.divider));
    NativeRenderer::set_attr(&row, "data-pad",           "10 16 10 14");
    NativeRenderer::append(parent, &row);

    let icon_sz: u32 = match icon_style {
        EntryIconStyle::Champion | EntryIconStyle::Item | EntryIconStyle::Rune => 128,
        EntryIconStyle::None => 0,
    };
    if icon_sz > 0 {
        let icon_path: Option<String> = match icon_style {
            EntryIconStyle::Champion => {
                ALL_CHAMPIONS.iter().copied()
                    .find(|&c| name_slug(c) == name_slug(ddragon_key(&change.patch)))
                    .map(|key| format!("assets/champion_icons/{key}.png"))
            }
            EntryIconStyle::Item => {
                item_icon_id(&change.patch).map(|id| format!("assets/item_icons/{id}.png"))
            }
            EntryIconStyle::Rune => {
                perk_icon_id(&change.patch).map(|id| format!("assets/rune_icons/{id}.png"))
            }
            EntryIconStyle::None => None,
        };
        let icon = NativeRenderer::element("div");
        NativeRenderer::set_attr(&icon, "data-w",             &icon_sz.to_string());
        NativeRenderer::set_attr(&icon, "data-height",        &icon_sz.to_string());
        NativeRenderer::set_attr(&icon, "data-border-top",    &format!("{}:2", badge_color));
        NativeRenderer::set_attr(&icon, "data-border-left",   &format!("{}:2", badge_color));
        NativeRenderer::set_attr(&icon, "data-border-bottom", &format!("{}:2", badge_color));
        NativeRenderer::set_attr(&icon, "data-border-right",  &format!("{}:2", badge_color));
        match icon_path {
            Some(p) => NativeRenderer::set_attr(&icon, "data-image", &p),
            None    => NativeRenderer::set_attr(&icon, "data-fill",  badge_color),
        }
        NativeRenderer::append(&row, &icon);
        let gap = NativeRenderer::element("div");
        NativeRenderer::set_attr(&gap, "data-w",      "14");
        NativeRenderer::set_attr(&gap, "data-height", "1");
        NativeRenderer::append(&row, &gap);
    }

    let col = NativeRenderer::element("div");
    NativeRenderer::set_attr(&col, "data-layout", "column");
    NativeRenderer::set_attr(&col, "data-flex",   "1.0");
    NativeRenderer::append(&row, &col);

    // Combine name + badge into one text node so they are truly adjacent —
    // separate nodes require a fixed data-w on the name, which leaves a gap
    // for short names. Combined text fills the row naturally with no gap.
    let display  = format!("{}{}", change.patch, badge_label);
    let name_col = if badge_label.is_empty() { theme.text } else { badge_color };
    let name_t   = NativeRenderer::text(&display);
    NativeRenderer::set_attr(&name_t, "data-color",       name_col);
    NativeRenderer::set_attr(&name_t, "data-text-size",   "20");
    NativeRenderer::set_attr(&name_t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&name_t, "data-height",      "36");
    NativeRenderer::append(&col, &name_t);

    for line in &summary_lines {
        let lt = NativeRenderer::text(line);
        NativeRenderer::set_attr(&lt, "data-color",     theme.muted);
        NativeRenderer::set_attr(&lt, "data-text-size", "15");
        NativeRenderer::set_attr(&lt, "data-height",    "22");
        NativeRenderer::append(&col, &lt);
    }
}

// ── State ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tab { Champions, Runes, PatchNotes, Automation, Status }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PatchDetailTab { Champions, Items, Runes, System }

#[derive(Clone, Debug)]
pub enum RuneStatus {
    Idle,
    Applying,
    Applied(String),
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoleFilter { All, Top, Jng, Mid, Bot, Sup }

impl RoleFilter {
    fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Top => "Top",
            Self::Jng => "Jng",
            Self::Mid => "Mid",
            Self::Bot => "Bot",
            Self::Sup => "Sup",
        }
    }
}

/// Message sent from UI click handlers to the async rune worker.
pub enum RuneCmd {
    Apply(RuneRecommendation),
}

/// Message sent from champion-select click handlers to the async fetch worker.
pub enum FetchCmd {
    /// Fetch the live build + patch history for this champion.
    All(String),
    /// Fetch and store the current patch version string (e.g. `"25.12"`).
    PatchVersion,
    /// Fetch all champion changes from the latest patch for the global overview.
    GlobalNotes,
    /// Re-fetch patch notes using a different lookback depth, then re-run any
    /// pending champion history fetch.  Sent when the user adjusts the patch
    /// window control in the Patch Notes tab.
    ReloadNotes { depth: u32, champion: Option<String> },
    /// Download rune icons from Data Dragon into assets/rune_icons/{id}.png.
    RuneIcons,
}

#[derive(Clone)]
pub struct ControlPanelState {
    pub active_tab:        Tab,
    pub selected_champion: Option<String>,
    pub current_build:     Option<BuildRecommendation>,
    pub rune_status:       RuneStatus,
    /// Which of the up-to-3 recommended rune pages is currently displayed (0-based).
    pub selected_rune_page: usize,
    pub game_active:       bool,
    pub recommender:       Arc<LiveRecommender>,
    pub rune_tx:           std::sync::mpsc::SyncSender<RuneCmd>,
    pub fetch_tx:          std::sync::mpsc::SyncSender<FetchCmd>,
    /// Champion patch changes — populated async by the fetch worker.
    pub patch_changes:     Vec<PatchChange>,
    pub patch_loading:     bool,
    /// Current LoL patch version string, e.g. `"25.12"` — populated on startup.
    pub current_patch:     Option<String>,
    /// Champion changes grouped by patch, newest-first — populated via GlobalNotes/ReloadNotes.
    pub global_patches:    Vec<(String, Vec<PatchChange>)>,
    pub global_loading:    bool,
    /// Champion role filter for the picker grid.
    pub champ_role_filter: RoleFilter,
    /// Live search query typed by the user (lowercased).
    pub search_query:      String,
    /// Shared with the automation background thread — writes here take effect
    /// on the next 500 ms poll tick without restarting anything.
    pub automation:        Arc<Mutex<AutomationConfig>>,
    /// Patch note blocks that are currently expanded (empty = all collapsed).
    pub patch_expanded:    HashSet<String>,
    /// Number of recent patches to show in the champion patch history (default 3).
    pub patch_lookback:    u32,
    /// When `Some((role, is_ban))`, the Automation tab shows a full-panel
    /// champion picker for that role slot instead of the normal controls.
    /// `role` is the LCU position key: `"top"/"jungle"/"middle"/"bottom"/"utility"`.
    pub configuring_slot:  Option<(String, bool)>,
    /// Which patch is selected in the patch notes sidebar (None = most recent).
    pub selected_patch_ver: Option<String>,
    /// Which detail tab is active inside the patch notes view.
    pub patch_detail_tab:   PatchDetailTab,
}

impl ControlPanelState {
    pub fn new(
        recommender: Arc<LiveRecommender>,
        rune_tx:     std::sync::mpsc::SyncSender<RuneCmd>,
        fetch_tx:    std::sync::mpsc::SyncSender<FetchCmd>,
        automation:  Arc<Mutex<AutomationConfig>>,
    ) -> Self {
        Self {
            active_tab:        Tab::Runes,
            selected_champion: None,
            current_build:      None,
            rune_status:        RuneStatus::Idle,
            selected_rune_page: 0,
            game_active:       false,
            recommender,
            rune_tx,
            fetch_tx,
            patch_changes:     vec![],
            patch_loading:     false,
            current_patch:     None,
            global_patches:    vec![],
            global_loading:    false,
            champ_role_filter: RoleFilter::All,
            search_query:      String::new(),
            automation,
            patch_expanded:    HashSet::new(),
            patch_lookback:    3,
            configuring_slot:  None,
            selected_patch_ver: None,
            patch_detail_tab:   PatchDetailTab::Champions,
        }
    }
}

// ── Rune worker ──────────────────────────────────────────────────────────────

/// Spawn a background thread that receives [`RuneCmd`]s and applies rune pages
/// to the live League Client.  Updates `state.rune_status` when done.
pub fn spawn_rune_worker(
    rx:        std::sync::mpsc::Receiver<RuneCmd>,
    state_arc: Arc<Mutex<ControlPanelState>>,
) {
    std::thread::Builder::new()
        .name("rune-worker".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rune worker runtime");
            for cmd in rx {
                let RuneCmd::Apply(runes) = cmd;
                let name = runes.name.clone();
                let result = rt.block_on(async {
                    let lf = cathedral_rift::Lockfile::discover()?;
                    let client = cathedral_rift::LcuClient::from_lockfile(&lf)?;
                    client.apply_rune_page(&runes).await
                });
                let status = match result {
                    Ok(()) => {
                        tracing::info!(runes = %name, "rune page applied via LCU");
                        RuneStatus::Applied(format!("Applied: {name}"))
                    }
                    Err(e) => {
                        tracing::warn!(err = %e, "rune page apply failed");
                        RuneStatus::Error(format!("LCU error: {e}"))
                    }
                };
                if let Ok(mut s) = state_arc.lock() {
                    s.rune_status = status;
                }
            }
        })
        .expect("spawn rune-worker");
}

// ── Fetch worker ─────────────────────────────────────────────────────────────

/// Spawn a background thread that receives [`FetchCmd`]s, fetches live rune
/// recommendations and patch change history, then writes results back to state.
pub fn spawn_fetch_worker(
    rx:        std::sync::mpsc::Receiver<FetchCmd>,
    state_arc: Arc<Mutex<ControlPanelState>>,
) {
    std::thread::Builder::new()
        .name("fetch-worker".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("fetch worker runtime");
            for cmd in rx {
                // Grab the recommender handle without holding the lock during fetch.
                let rec = state_arc.lock().ok().map(|s| Arc::clone(&s.recommender));
                let Some(rec) = rec else { continue };

                match cmd {
                    FetchCmd::PatchVersion => {
                        let version = rt.block_on(rec.current_patch_string());
                        if let Ok(mut s) = state_arc.lock() {
                            s.current_patch = Some(version);
                        }
                    }
                    FetchCmd::GlobalNotes => {
                        if let Ok(mut s) = state_arc.lock() { s.global_loading = true; }
                        let grouped = rt.block_on(rec.fetch_all_changes_grouped());
                        // Collect item IDs from patch changes so their icons are cached.
                        let patch_item_ids: Vec<u32> = grouped.iter()
                            .flat_map(|(_, changes)| changes.iter())
                            .filter_map(|c| item_icon_id(&c.patch))
                            .collect::<std::collections::HashSet<_>>()
                            .into_iter()
                            .collect();
                        if let Ok(mut s) = state_arc.lock() {
                            s.global_patches = grouped;
                            s.global_loading  = false;
                        }
                        if !patch_item_ids.is_empty() {
                            let assets_dir = std::env::current_exe()
                                .ok()
                                .and_then(|p| p.parent().map(|d| d.join("assets")))
                                .unwrap_or_else(|| std::path::PathBuf::from("assets"));
                            let dd = cathedral_rift::DataDragonClient::new();
                            let _ = rt.block_on(dd.download_item_icons(&assets_dir, &patch_item_ids));
                        }
                    }
                    FetchCmd::All(champion) => {
                        let champion_ref = champion.as_str();
                        let (build, changes) = rt.block_on(async {
                            tokio::join!(
                                rec.fetch_and_cache(champion_ref),
                                rec.fetch_changes(champion_ref),
                            )
                        });

                        let item_ids        = build.items.item_ids.clone();
                        let summoner_spells = build.summoner_spells.clone();
                        let mut stored = false;
                        if let Ok(mut s) = state_arc.lock() {
                            // Only update if the same champion is still selected
                            // (user may have clicked a different one in flight).
                            if s.selected_champion.as_deref() == Some(&champion) {
                                s.current_build = Some(build);
                                s.patch_changes = changes;
                                s.patch_loading = false;
                                stored = true;
                            }
                        }
                        if stored {
                            let assets_dir = std::env::current_exe()
                                .ok()
                                .and_then(|p| p.parent().map(|d| d.join("assets")))
                                .unwrap_or_else(|| std::path::PathBuf::from("assets"));
                            let dd = cathedral_rift::DataDragonClient::new();
                            rt.block_on(async {
                                let _ = tokio::join!(
                                    dd.download_item_icons(&assets_dir, &item_ids),
                                    dd.download_summoner_icons(&assets_dir, &summoner_spells),
                                );
                            });
                        }
                    }
                    FetchCmd::ReloadNotes { depth, champion } => {
                        // Mark loading state.
                        if let Ok(mut s) = state_arc.lock() {
                            s.global_loading = true;
                            if champion.is_some() { s.patch_loading = true; }
                        }
                        // Re-fetch patch notes cache at the new depth.
                        rt.block_on(rec.load_patch_notes_n(depth as usize));
                        // Refresh global overview at new depth.
                        let grouped = rt.block_on(rec.fetch_all_changes_grouped());
                        if let Ok(mut s) = state_arc.lock() {
                            s.global_patches = grouped;
                            s.global_loading  = false;
                        }
                        // Refresh champion history if one is selected.
                        if let Some(champ) = champion {
                            let changes = rt.block_on(rec.fetch_changes(&champ));
                            if let Ok(mut s) = state_arc.lock() {
                                if s.selected_champion.as_deref() == Some(&champ) {
                                    s.patch_changes  = changes;
                                    s.patch_loading   = false;
                                }
                            }
                        }
                    }
                    FetchCmd::RuneIcons => {
                        let assets_dir = std::env::current_exe()
                            .ok()
                            .and_then(|p| p.parent().map(|d| d.join("assets")))
                            .unwrap_or_else(|| std::path::PathBuf::from("assets"));
                        let dd = cathedral_rift::DataDragonClient::new();
                        match rt.block_on(dd.download_rune_icons(&assets_dir)) {
                            Ok(n) if n > 0 => tracing::info!(n, "rune icons downloaded"),
                            Ok(_)          => tracing::debug!("rune icons already up to date"),
                            Err(e)         => tracing::warn!(%e, "rune icon download failed"),
                        }
                    }
                }
            }
        })
        .expect("spawn fetch-worker");
}

// ── Public render API ────────────────────────────────────────────────────────

pub fn build_control_panel(state_arc: Arc<Mutex<ControlPanelState>>) -> NativeNode {
    let root = NativeRenderer::element("div");
    render_control_panel_into(&root, state_arc, PANEL_W, PANEL_H);
    root
}

/// Called from `on_pre_paint` every frame — clears and rebuilds the scene.
pub fn render_control_panel_into(
    root:      &NativeNode,
    state_arc: Arc<Mutex<ControlPanelState>>,
    win_w:     u32,
    win_h:     u32,
) {
    // Process keyboard input for champion search.
    let chars = NativeRenderer::take_pending_chars();
    if !chars.is_empty() {
        if let Ok(mut s) = state_arc.lock() {
            for c in chars.chars() {
                match c {
                    '\x08' => { s.search_query.pop(); }    // Backspace
                    '\x1b' => { s.search_query.clear(); }  // Escape
                    '\t'   => {  // Tab — toggle champion picker
                        s.active_tab = if s.active_tab == Tab::Champions {
                            Tab::Runes
                        } else {
                            Tab::Champions
                        };
                        s.search_query.clear();
                    }
                    '\x1c' => {  // ArrowLeft
                        if s.active_tab == Tab::Runes {
                            s.selected_rune_page = s.selected_rune_page.saturating_sub(1);
                        } else if s.active_tab == Tab::PatchNotes {
                            const DT: &[PatchDetailTab] = &[PatchDetailTab::Champions, PatchDetailTab::Items, PatchDetailTab::Runes, PatchDetailTab::System];
                            let idx = DT.iter().position(|&t| t == s.patch_detail_tab).unwrap_or(0);
                            s.patch_detail_tab = DT[(idx + DT.len() - 1) % DT.len()];
                        }
                    }
                    '\x1d' => {  // ArrowRight
                        if s.active_tab == Tab::Runes {
                            s.selected_rune_page = (s.selected_rune_page + 1).min(2);
                        } else if s.active_tab == Tab::PatchNotes {
                            const DT: &[PatchDetailTab] = &[PatchDetailTab::Champions, PatchDetailTab::Items, PatchDetailTab::Runes, PatchDetailTab::System];
                            let idx = DT.iter().position(|&t| t == s.patch_detail_tab).unwrap_or(0);
                            s.patch_detail_tab = DT[(idx + 1) % DT.len()];
                        }
                    }
                    '\x1e' => {  // ArrowUp — previous tab
                        const TABS: &[Tab] = &[Tab::Champions, Tab::Runes, Tab::PatchNotes, Tab::Automation, Tab::Status];
                        let idx = TABS.iter().position(|&t| t == s.active_tab).unwrap_or(0);
                        s.active_tab = TABS[(idx + TABS.len() - 1) % TABS.len()];
                    }
                    '\x1f' => {  // ArrowDown — next tab
                        const TABS: &[Tab] = &[Tab::Champions, Tab::Runes, Tab::PatchNotes, Tab::Automation, Tab::Status];
                        let idx = TABS.iter().position(|&t| t == s.active_tab).unwrap_or(0);
                        s.active_tab = TABS[(idx + 1) % TABS.len()];
                    }
                    '\r' | '\n' => {  // Enter — apply runes when on the Runes tab
                        if s.active_tab == Tab::Runes && !matches!(s.rune_status, RuneStatus::Applying) {
                            let page = s.current_build.as_ref().and_then(|build| {
                                let pages: Vec<&RuneRecommendation> = std::iter::once(&build.runes)
                                    .chain(build.alt_runes.iter())
                                    .take(3)
                                    .collect();
                                let idx = s.selected_rune_page.min(pages.len().saturating_sub(1));
                                pages.get(idx).map(|p| (*p).clone())
                            });
                            if let Some(page) = page {
                                s.rune_status = RuneStatus::Applying;
                                let _ = s.rune_tx.try_send(RuneCmd::Apply(page));
                            }
                        }
                    }
                    c if c.is_alphabetic() => { s.search_query.push(c.to_ascii_lowercase()); }
                    _ => {}
                }
            }
        }
    }

    let state = state_arc.lock().map(|g| g.clone()).unwrap_or_else(|p| p.into_inner().clone());
    let theme = Theme::default();

    NativeRenderer::remove_node(root);
    NativeRenderer::set_attr(root, "data-fill", theme.bg);

    NativeRenderer::append(root, &titlebar_panel(&state, &theme, win_w));
    NativeRenderer::append(root, &sidebar_panel(&state, state_arc.clone(), &theme, win_h));
    NativeRenderer::append(root, &main_panel(&state, state_arc, &theme, win_w, win_h));
}

// ── Title bar (custom; decorations: false) ────────────────────────────────────
//
// Layout (all flex fractions sum to 1.0):
//   drag zone (0.73)  logo(0.04) + title(0.30) + tag(0.66)
//   status    (0.18)
//   min / max / close (0.03 each)

fn titlebar_panel(state: &ControlPanelState, theme: &Theme, win_w: u32) -> NativeNode {
    let p = abs_panel(0, 0, win_w, TITLEBAR_H, theme.surface);
    NativeRenderer::set_attr(&p, "data-border-bottom", &format!("{}:2", theme.accent));

    let row = hrow(TITLEBAR_H);

    // Drag zone — left 70 %; clicking here moves the window.
    // The tag cell (0.66 inside) soaks up the empty draggable space after the title.
    let drag = NativeRenderer::element("div");
    NativeRenderer::set_attr(&drag, "data-drag-window", "true");
    NativeRenderer::set_attr(&drag, "data-layout", "row");
    NativeRenderer::set_attr(&drag, "data-flex", "0.73");
    NativeRenderer::set_attr(&drag, "data-height", &TITLEBAR_H.to_string());
    NativeRenderer::set_attr(&drag, "data-pad", "0 0 0 14");

    let logo = NativeRenderer::element("div");
    NativeRenderer::set_attr(&logo, "data-image", "assets/IH_Icon_simple.png");
    NativeRenderer::set_attr(&logo, "data-flex", "0.04");
    NativeRenderer::set_attr(&logo, "data-height", &TITLEBAR_H.to_string());
    NativeRenderer::append(&drag, &logo);

    let title = NativeRenderer::text("INGENIOUS HUNTER");
    NativeRenderer::set_attr(&title, "data-color", theme.accent);
    NativeRenderer::set_attr(&title, "data-text-size", "16");
    NativeRenderer::set_attr(&title, "data-text-weight", "bold");
    NativeRenderer::set_attr(&title, "data-flex", "0.30");
    NativeRenderer::set_attr(&title, "data-height", &TITLEBAR_H.to_string());
    NativeRenderer::set_attr(&title, "data-pad", &format!("{} 0 0 0", (TITLEBAR_H - 16) / 2));
    NativeRenderer::append(&drag, &title);

    // Tag fills the rest of the drag zone (all empty = draggable).
    let tag = NativeRenderer::text(&format!("[{}]", BUILD));
    NativeRenderer::set_attr(&tag, "data-color", theme.muted);
    NativeRenderer::set_attr(&tag, "data-text-size", "14");
    NativeRenderer::set_attr(&tag, "data-flex", "0.66"); // 0.04 + 0.30 + 0.66 = 1.0
    NativeRenderer::set_attr(&tag, "data-height", &TITLEBAR_H.to_string());
    NativeRenderer::append(&drag, &tag);

    NativeRenderer::append(&row, &drag);

    // Status badge
    let (badge_text, badge_color) = if state.game_active {
        ("● GAME ACTIVE", theme.ok)
    } else {
        ("○ STANDBY", theme.muted)
    };
    let badge = NativeRenderer::text(badge_text);
    NativeRenderer::set_attr(&badge, "data-color", badge_color);
    NativeRenderer::set_attr(&badge, "data-text-size", "14");
    NativeRenderer::set_attr(&badge, "data-flex", "0.18"); // 0.73 + 0.18 + 3*0.03 = 1.0
    NativeRenderer::set_attr(&badge, "data-height", &TITLEBAR_H.to_string());
    NativeRenderer::append(&row, &badge);

    // Window controls: minimize · maximize · close  (0.05 each)
    NativeRenderer::append(&row, &win_ctrl_btn("─", false, theme, WindowCmd::Minimize));
    NativeRenderer::append(&row, &win_ctrl_btn("□", false, theme, WindowCmd::ToggleMaximize));
    NativeRenderer::append(&row, &win_ctrl_btn("✕", true,  theme, WindowCmd::Close));

    NativeRenderer::append(&p, &row);
    p
}

fn win_ctrl_btn(symbol: &str, is_close: bool, theme: &Theme, cmd: WindowCmd) -> NativeNode {
    let btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&btn, "data-flex", "0.03");
    NativeRenderer::set_attr(&btn, "data-height", &TITLEBAR_H.to_string());
    NativeRenderer::set_attr(&btn, "data-hover-fill",
        if is_close { "#A01820" } else { theme.surface_hi });

    let t = NativeRenderer::text(symbol);
    NativeRenderer::set_attr(&t, "data-color", if is_close { "#ff6070" } else { theme.muted });
    NativeRenderer::set_attr(&t, "data-text-size", "18");
    NativeRenderer::set_attr(&t, "data-height", &TITLEBAR_H.to_string());
    // Pad top centres an 18px glyph in TITLEBAR_H: (64 - 18) / 2 = 23
    NativeRenderer::set_attr(&t, "data-pad", &format!("{} 0 0 0", (TITLEBAR_H - 18) / 2));
    NativeRenderer::set_attr(&t, "data-align", "center");
    NativeRenderer::append(&btn, &t);

    NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
        NativeRenderer::request_window_cmd(cmd);
    }));

    btn
}

// ── Sidebar ───────────────────────────────────────────────────────────────────

fn sidebar_panel(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    win_h:     u32,
) -> NativeNode {
    let body_h = win_h.saturating_sub(TITLEBAR_H);
    let p = abs_panel(0, TITLEBAR_H as i32, SIDEBAR_W, body_h, theme.surface);
    NativeRenderer::set_attr(&p, "data-border-right", &format!("{}:1", theme.divider));

    spacer(16, &p);

    for (tab, label) in [
        (Tab::Champions,   "> CHAMPIONS"),
        (Tab::Runes,       "> RUNES"),
        (Tab::PatchNotes,  "> PATCH NOTES"),
        (Tab::Automation,  "> AUTOMATION"),
        (Tab::Status,      "> STATUS"),
    ] {
        NativeRenderer::append(
            &p,
            &tab_button(label, tab, state.active_tab == tab, state_arc.clone(), theme),
        );
    }

    p
}

fn tab_button(
    label:     &str,
    tab:       Tab,
    active:    bool,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
) -> NativeNode {
    // btn must be the hit target (text nodes return None from hit_test).
    let btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&btn, "data-height",      "48");
    NativeRenderer::set_attr(&btn, "data-fill",        if active { "#1e1020" } else { "transparent" });
    NativeRenderer::set_attr(&btn, "data-hover-fill",  theme.surface_hi);
    NativeRenderer::set_attr(&btn, "data-border-left", &format!("{}:4", if active { theme.accent } else { "transparent" }));
    NativeRenderer::set_attr(&btn, "data-pad",         "0 0 0 14");
    NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = state_arc.lock() { s.active_tab = tab; }
    }));
    let txt = NativeRenderer::text(label);
    NativeRenderer::set_attr(&txt, "data-color",     if active { theme.text } else { "#8888A8" });
    NativeRenderer::set_attr(&txt, "data-text-size", "14");
    NativeRenderer::set_attr(&txt, "data-height",    "48");
    NativeRenderer::append(&btn, &txt);
    btn
}

// ── Main panel ────────────────────────────────────────────────────────────────

fn main_panel(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    win_w:     u32,
    win_h:     u32,
) -> NativeNode {
    let main_w = win_w.saturating_sub(SIDEBAR_W);
    let body_h = win_h.saturating_sub(TITLEBAR_H);
    let p = abs_panel(SIDEBAR_W as i32, TITLEBAR_H as i32, main_w, body_h, theme.bg);
    NativeRenderer::set_attr(&p, "data-layout", "column");

    const BOTTOM_H: u32 = 110;
    let show_bottom = state.active_tab == Tab::Runes;
    let tab_h = if show_bottom { body_h.saturating_sub(BOTTOM_H) } else { body_h };

    let tab_area = NativeRenderer::element("div");
    NativeRenderer::set_attr(&tab_area, "data-layout", "column");
    NativeRenderer::set_attr(&tab_area, "data-w",      &main_w.to_string());
    NativeRenderer::set_attr(&tab_area, "data-height", &tab_h.to_string());
    NativeRenderer::append(&p, &tab_area);

    match state.active_tab {
        Tab::Champions  => champion_tab(state, state_arc.clone(), theme, &tab_area, main_w, tab_h),
        Tab::Runes      => rune_tab(state, state_arc.clone(), theme, &tab_area, main_w, tab_h),
        Tab::PatchNotes => patch_notes_tab(state, state_arc.clone(), theme, &tab_area, main_w, tab_h),
        Tab::Automation => automation_tab(state, state_arc.clone(), theme, &tab_area, tab_h),
        Tab::Status     => status_tab(state, theme, &tab_area),
    }

    if show_bottom {
        champ_select_bar(state, state_arc, theme, &p, main_w, BOTTOM_H);
    }

    p
}

// ── Runes tab ─────────────────────────────────────────────────────────────────

/// Champion header: thin bar on the left, splash in the top-right quadrant.
///
/// Layout (row):
///   left col (main_w/2 × hdr_h):  transparent bar with icon + name + role
/// Thin champion bar (icon + name/role + CHANGE CHAMPION button).
/// Splash art is handled as a data-overlay in `rune_tab` — this function
/// only emits the BAR_H-tall flow element.
///
/// Returns the height consumed (BAR_H = 72).
fn rune_champ_header(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    main_w:    u32,
) -> u32 {
    const BAR_H:   u32 = 288;
    const ICON_SZ: u32 = 240;
    const ICON_PAD: u32 = (BAR_H - ICON_SZ) / 2;

    let Some(name) = &state.selected_champion else { return 0 };

    let bar = NativeRenderer::element("div");
    NativeRenderer::set_attr(&bar, "data-layout", "row");
    NativeRenderer::set_attr(&bar, "data-w",      &main_w.to_string());
    NativeRenderer::set_attr(&bar, "data-height", &BAR_H.to_string());
    NativeRenderer::append(parent, &bar);

    // Icon — equally padded on all sides, square
    let icon_wrapper = NativeRenderer::element("div");
    NativeRenderer::set_attr(&icon_wrapper, "data-layout", "column");
    NativeRenderer::set_attr(&icon_wrapper, "data-w",      &(ICON_SZ + ICON_PAD * 2).to_string());
    NativeRenderer::set_attr(&icon_wrapper, "data-height", &BAR_H.to_string());
    NativeRenderer::set_attr(&icon_wrapper, "data-pad",    &format!("{ICON_PAD} {ICON_PAD} {ICON_PAD} {ICON_PAD}"));
    NativeRenderer::append(&bar, &icon_wrapper);

    let icon = NativeRenderer::element("div");
    NativeRenderer::set_attr(&icon, "data-w",             &ICON_SZ.to_string());
    NativeRenderer::set_attr(&icon, "data-height",        &ICON_SZ.to_string());
    NativeRenderer::set_attr(&icon, "data-image",         &format!("assets/champion_icons/{name}.png"));
    NativeRenderer::set_attr(&icon, "data-border-top",    &format!("{}:2", theme.accent));
    NativeRenderer::set_attr(&icon, "data-border-bottom", &format!("{}:2", theme.accent));
    NativeRenderer::set_attr(&icon, "data-border-left",   &format!("{}:2", theme.accent));
    NativeRenderer::set_attr(&icon, "data-border-right",  &format!("{}:2", theme.accent));
    NativeRenderer::append(&icon_wrapper, &icon);

    // Name + role — vertically centred in bar
    let name_h: u32 = 56;
    let role_h: u32 = 36;
    let name_top = BAR_H.saturating_sub(name_h + 4 + role_h) / 2;

    let name_col = NativeRenderer::element("div");
    NativeRenderer::set_attr(&name_col, "data-layout", "column");
    NativeRenderer::set_attr(&name_col, "data-flex",   "1.0");
    NativeRenderer::set_attr(&name_col, "data-height", &BAR_H.to_string());
    NativeRenderer::set_attr(&name_col, "data-pad",    &format!("{name_top} 0 0 {ICON_PAD}"));
    NativeRenderer::append(&bar, &name_col);

    let name_lbl = NativeRenderer::text(name.as_str());
    NativeRenderer::set_attr(&name_lbl, "data-color",       theme.text);
    NativeRenderer::set_attr(&name_lbl, "data-text-size",   "44");
    NativeRenderer::set_attr(&name_lbl, "data-text-weight", "bold");
    NativeRenderer::set_attr(&name_lbl, "data-h",           &name_h.to_string());
    NativeRenderer::append(&name_col, &name_lbl);

    spacer(4, &name_col);

    let role_lbl = NativeRenderer::text(champion_primary_role(name.as_str()));
    NativeRenderer::set_attr(&role_lbl, "data-color",     theme.accent2);
    NativeRenderer::set_attr(&role_lbl, "data-text-size", "28");
    NativeRenderer::set_attr(&role_lbl, "data-h",         &role_h.to_string());
    NativeRenderer::append(&name_col, &role_lbl);

    // CHANGE CHAMPION button — data-overlay, horizontally aligned with the items
    // card column (same x and width as used in build_preview's side-by-side layout),
    // vertically centred within BAR_H.
    const BTN_H: u32 = 72;
    let side_pad = 12u32;
    let gap      = 8u32;
    let items_fw = 364u32; // items_w(360) + border(4)
    let rune_w   = main_w.saturating_sub(side_pad + gap + items_fw + 4).min(1060).max(500);
    let rune_fw  = rune_w + 4;
    let btn_x    = SIDEBAR_W as i32 + side_pad as i32 + rune_fw as i32 + gap as i32;
    let btn_w    = items_fw;
    let btn_y    = TITLEBAR_H as i32 + (BAR_H as i32 - BTN_H as i32) / 2;

    let change_btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&change_btn, "data-overlay",       "true");
    NativeRenderer::set_attr(&change_btn, "data-x",             &btn_x.to_string());
    NativeRenderer::set_attr(&change_btn, "data-y",             &btn_y.to_string());
    NativeRenderer::set_attr(&change_btn, "data-w",             &btn_w.to_string());
    NativeRenderer::set_attr(&change_btn, "data-h",             &BTN_H.to_string());
    NativeRenderer::set_attr(&change_btn, "data-fill",          "#200a10");
    NativeRenderer::set_attr(&change_btn, "data-hover-fill",    theme.accent);
    NativeRenderer::set_attr(&change_btn, "data-border-top",    &format!("{}:3", theme.accent));
    NativeRenderer::set_attr(&change_btn, "data-border-bottom", &format!("{}:3", theme.accent));
    NativeRenderer::set_attr(&change_btn, "data-border-left",   &format!("{}:3", theme.accent));
    NativeRenderer::set_attr(&change_btn, "data-border-right",  &format!("{}:3", theme.accent));
    NativeRenderer::append(parent, &change_btn);

    let change_lbl = NativeRenderer::text("CHANGE CHAMPION");
    NativeRenderer::set_attr(&change_lbl, "data-color",       theme.text);
    NativeRenderer::set_attr(&change_lbl, "data-text-size",   "18");
    NativeRenderer::set_attr(&change_lbl, "data-text-weight", "bold");
    NativeRenderer::set_attr(&change_lbl, "data-align",       "center");
    NativeRenderer::set_attr(&change_lbl, "data-h",           &BTN_H.to_string());
    NativeRenderer::set_attr(&change_lbl, "data-pad",         &format!("{} 0 0 0", (BTN_H - 22) / 2));
    NativeRenderer::append(&change_btn, &change_lbl);

    let sa = state_arc.clone();
    NativeRenderer::on_event(&change_btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = sa.lock() { s.active_tab = Tab::Champions; }
    }));

    BAR_H
}

/// Static perk/rune name lookup by Riot ID.  Covers all keystones, common
/// runes, and shards.  Returns `""` for unknown IDs.
fn perk_name(id: u32) -> &'static str {
    match id {
        // Precision
        8005 => "Press the Attack",   8008 => "Lethal Tempo",
        8010 => "Conqueror",          8021 => "Fleet Footwork",
        9101 => "Absorb Life",        9111 => "Triumph",            8009 => "Presence of Mind",
        9104 => "Legend: Alacrity",   9105 => "Legend: Haste",      9103 => "Legend: Bloodline",
        8014 => "Coup de Grace",      8017 => "Cut Down",           8299 => "Last Stand",
        // Domination
        8112 => "Electrocute",        8128 => "Dark Harvest",       9923 => "Hail of Blades",
        8126 => "Cheap Shot",         8139 => "Taste of Blood",     8143 => "Sudden Impact",
        8137 => "Sixth Sense",        8140 => "Grisly Mementos",    8141 => "Deep Ward",
        8135 => "Treasure Hunter",    8105 => "Relentless Hunter",  8106 => "Ultimate Hunter",
        // Sorcery
        8214 => "Summon Aery",        8229 => "Arcane Comet",       8230 => "Phase Rush",
        8224 => "Axiom Arcanist",     8226 => "Manaflow Band",      8275 => "Nimbus Cloak",
        8210 => "Transcendence",      8234 => "Celerity",           8233 => "Absolute Focus",
        8237 => "Scorch",             8232 => "Waterwalking",       8236 => "Gathering Storm",
        // Inspiration
        8351 => "Glacial Augment",    8360 => "Unsealed Spellbook", 8369 => "First Strike",
        8306 => "Hextech Flashtraption", 8304 => "Magical Footwear", 8321 => "Cash Back",
        8313 => "Triple Tonic",       8352 => "Time Warp Tonic",    8345 => "Biscuit Delivery",
        8347 => "Cosmic Insight",     8410 => "Approach Velocity",  8316 => "Jack Of All Trades",
        // Resolve
        8437 => "Grasp of the Undying", 8439 => "Aftershock",      8465 => "Guardian",
        8446 => "Demolish",           8463 => "Font of Life",       8401 => "Shield Bash",
        8429 => "Conditioning",       8444 => "Second Wind",        8473 => "Bone Plating",
        8451 => "Overgrowth",         8453 => "Revitalize",         8242 => "Unflinching",
        // Shards
        5001 => "+15–90 HP",          5002 => "+6 Armor",           5003 => "+8 MR",
        5005 => "+10% AS",            5007 => "+8 Haste",           5008 => "+9 Adaptive",
        5010 => "+2% Move Speed",     5011 => "+65 HP",             5013 => "+10% Tenacity",
        _    => "",
    }
}

/// Full description for each perk — shown under the rune name.
/// Returns `""` for shards (their name already states the effect).
fn perk_desc(id: u32) -> &'static str {
    match id {
        // Precision — keystones
        8005 => "Hit an enemy 3 consecutive times to mark them: all sources deal +12% damage for 6s",
        8008 => "Stack attack speed by attacking champions; attack speed cap is removed while active",
        8010 => "Stack 12× adaptive force attacking champions; at max stacks heal 5–15% of damage dealt",
        8021 => "Build 100 Energize stacks moving and attacking; proc heals 3–94 HP and grants 20% move speed",
        // Precision — row 2
        9101 => "Killing a unit heals you for a small amount; tripled below 30% HP",
        9111 => "Takedowns restore 12% of your missing HP and grant you 20 bonus gold",
        8009 => "Takedowns restore 15% max mana or energy and permanently increase max mana",
        // Precision — row 3
        9104 => "Takedowns grant +1.5% attack speed per Legend stack, up to 10 stacks (+15% total)",
        9105 => "Takedowns grant +1 ability haste per Legend stack, up to 10 stacks (+15 haste total)",
        9103 => "Takedowns grant +1% lifesteal per Legend stack, up to 10 stacks (+8% lifesteal total)",
        // Precision — row 4
        8014 => "Deal +8% increased damage to champions who have less than 40% HP",
        8017 => "Deal 5–15% more damage to champions with more HP than you (scales with HP gap)",
        8299 => "Deal 5–11% more damage when you are below 60% HP; maximised at 30% HP",
        // Domination — keystones
        8112 => "Hitting with 3 separate attacks or abilities in 3 seconds deals 30–220 bonus adaptive damage",
        8128 => "Damaging a champion below 50% HP steals their soul for bonus damage; unlimited stacks",
        9923 => "Entering combat with a champion grants 110% bonus attack speed for your first 3 attacks",
        // Domination — row 2
        8126 => "Damaging an impaired enemy champion deals 10–45 bonus true damage (0.5s cooldown)",
        8139 => "Heal yourself for 18–35 HP when you damage an enemy champion (20 second cooldown)",
        8143 => "Gain +7 Lethality and +7 Magic Penetration for 4 seconds after dashing, leaping, or blinking",
        // Domination — row 3
        8137 => "After 3 min, automatically detect one nearby hidden ward every 90 seconds",
        8140 => "Takedowns grant stacking Trinket Haste and out-of-combat movement speed bonus",
        8141 => "Enemy wards placed in the jungle or river gain bonus HP and increased ward duration",
        // Domination — row 4
        8135 => "Earn extra gold the first time you take down each unique enemy champion this game",
        8105 => "Takedowns grant +8 out-of-combat move speed, stacking up to 5 times (+40 total)",
        8106 => "Reduce your ultimate's cooldown per unique champion takedown; stacks up to 5 times",
        // Sorcery — keystones
        8214 => "Attacks and abilities dispatch Aery — she pokes enemies or shields allies, then returns to you",
        8229 => "Damaging a champion with an ability calls a comet to their location; comet refunds 20% of its CD",
        8230 => "Hit the same champion 3 times in 3 seconds: gain a burst of move speed and 75% Slow Resistance",
        // Sorcery — row 2
        8224 => "Your ultimate gains bonus damage or healing/shielding power; takedowns cut remaining cooldown by 15%",
        8226 => "Ability hits grow your max mana by 25 (cap 250 bonus mana); at cap restore 1% missing mana per second",
        8275 => "Casting a summoner spell grants a brief ghost-like burst of movement speed through units",
        // Sorcery — row 3
        8210 => "Gain +5 ability haste; for each point of haste past your cap, gain 1.5 adaptive force",
        8234 => "All movement speed bonuses are 7% stronger; 10% of all bonus move speed converts to AP or AD",
        8233 => "Gain +18 adaptive force (AP or AD) while you are above 70% of your maximum HP",
        // Sorcery — row 4
        8237 => "Your first ability hit every 10 seconds burns the enemy, dealing 15–35 bonus damage after 1 second",
        8232 => "While in the river gain +8% movement speed and +18 adaptive force (AP or AD)",
        8236 => "Gain +8 adaptive force every 10 minutes — no cap, scaling throughout the game",
        // Inspiration — keystones
        8351 => "First attack on a champion slows them; some triggered items also create frozen slowing zones",
        8360 => "Swap one of your summoner spells out of combat; swapped spells temporarily have shorter cooldowns",
        8369 => "First champion hit within 0.25s of entering combat: deal +10% damage for 3 seconds and earn gold",
        // Inspiration — row 2
        8306 => "While Flash is on cooldown it is replaced by Hexflash: channel to blink anywhere (60s cooldown)",
        8304 => "Receive free Slightly Magical Footwear at 12 minutes (250 gold value); each takedown saves 45 seconds",
        8321 => "Receive 100 gold back whenever you complete the purchase of a Legendary item",
        // Inspiration — row 3
        8313 => "Receive a free Combat Elixir at levels 3, 6, and 9 — one per threshold, not stockpiled",
        8352 => "On using a potion or elixir, immediately gain 50% of its healing effect; the rest ticks over time",
        8345 => "Receive a free Biscuit every 3 minutes until 12 minutes; each biscuit permanently increases max mana",
        // Inspiration — row 4
        8347 => "Gain +10 summoner spell ability haste and +10 item ability haste",
        8410 => "Gain +10% movement speed toward nearby allied champions who are impaired or enemies you have immobilised",
        8316 => "Gain +1 ability haste per unique stat type on your items — up to 10 different stats, up to +10 haste",
        // Resolve — keystones
        8437 => "Every 4s in combat your next attack heals you for 1.3% max HP, deals bonus magic damage, and grows max HP by 5",
        8439 => "Immobilising an enemy champion grants +40 armor and +40 MR for 2.5s; afterwards deal AoE magic damage nearby",
        8465 => "Shield an ally champion to share a protective shield with them; both receive it when standing near each other",
        // Resolve — row 2
        8446 => "Charge a powerful attack while near a tower (3s); deals 100 + 40% max HP bonus damage to it (100s cooldown)",
        8463 => "Impairing a champion marks them; allied attacks on the marked target heal all nearby allies over 2 seconds",
        8401 => "When you gain any shield: your next basic attack deals bonus damage equal to 1–5% of the shield + 1.5% bonus HP",
        // Resolve — row 3
        8429 => "After 12 minutes gain +9 Armor and +9 Magic Resistance; then increase both by an additional 5%",
        8444 => "After taking damage from an enemy champion: restore 6 HP plus 4% of your missing HP over 10 seconds",
        8473 => "After taking damage from a champion: the next 3 attacks or spells against you deal 20–60 less damage",
        // Resolve — row 4
        8451 => "Permanently gain 3 max HP for every 8 nearby enemy deaths; at 120 stacks gain 3.5% of max HP as bonus HP",
        8453 => "Your outgoing heals and shields are 5% stronger; if the target is below 40% HP they are 10% stronger instead",
        8242 => "Gain 10–25% Tenacity and Slow Resistance based on missing HP; increases further while you are crowd controlled",
        _    => "",
    }
}

/// Primary role for each champion, used by the role-filter chips in the picker.
fn champion_primary_role(name: &str) -> &'static str {
    match name {
        // Top
        "Aatrox" | "Ambessa" | "Camille" | "Chogath" | "Darius" | "DrMundo" | "Fiora"
        | "Gangplank" | "Garen" | "Gnar" | "Gwen" | "Illaoi" | "Irelia" | "Jax" | "Jayce"
        | "KSante" | "Kayle" | "Kennen" | "Kled" | "Malphite" | "Mordekaiser" | "Nasus"
        | "Olaf" | "Ornn" | "Pantheon" | "Poppy" | "Quinn" | "Renekton" | "Riven"
        | "Rumble" | "Sett" | "Shen" | "Singed" | "Sion" | "Teemo" | "Tryndamere"
        | "Urgot" | "Volibear" | "Yorick" => "Top",

        // Jungle
        "Amumu" | "Belveth" | "Briar" | "Diana" | "Ekko" | "Elise" | "Evelynn"
        | "FiddleSticks" | "Gragas" | "Graves" | "Hecarim" | "Ivern" | "JarvanIV"
        | "Karthus" | "Kayn" | "Khazix" | "Kindred" | "LeeSin" | "Lillia" | "MasterYi"
        | "MonkeyKing" | "Nidalee" | "Nocturne" | "Nunu" | "Rammus" | "RekSai" | "Rengar"
        | "Sejuani" | "Shaco" | "Shyvana" | "Skarner" | "Sylas" | "Taliyah" | "Trundle"
        | "Udyr" | "Vi" | "Viego" | "Warwick" | "XinZhao" | "Zac" => "Jng",

        // Mid
        "Ahri" | "Akali" | "Akshan" | "Anivia" | "Annie" | "AurelionSol" | "Aurora"
        | "Azir" | "Cassiopeia" | "Corki" | "Fizz" | "Galio" | "Heimerdinger" | "Hwei"
        | "Kassadin" | "Katarina" | "Leblanc" | "Lissandra" | "Malzahar" | "Mel"
        | "Naafiri" | "Neeko" | "Orianna" | "Qiyana" | "Ryze" | "Syndra" | "Talon"
        | "TwistedFate" | "Veigar" | "Vex" | "Viktor" | "Vladimir" | "Yasuo" | "Yone"
        | "Yunara" | "Zaahen" | "Zed" | "Ziggs" | "Zoe" => "Mid",

        // Bot / ADC
        "Aphelios" | "Ashe" | "Caitlyn" | "Draven" | "Ezreal" | "Jhin" | "Jinx" | "Kaisa"
        | "Kalista" | "KogMaw" | "Lucian" | "MissFortune" | "Nilah" | "Samira" | "Senna"
        | "Seraphine" | "Sivir" | "Smolder" | "Tristana" | "Twitch" | "Varus" | "Vayne"
        | "Xayah" | "Zeri" => "Bot",

        // Support
        "Alistar" | "Bard" | "Blitzcrank" | "Brand" | "Braum" | "Janna" | "Karma"
        | "Leona" | "Lulu" | "Lux" | "Maokai" | "Milio" | "Morgana" | "Nami" | "Nautilus"
        | "Pyke" | "Rakan" | "Rell" | "Renata" | "Sona" | "Soraka" | "Swain"
        | "TahmKench" | "Taric" | "Thresh" | "Velkoz" | "Xerath" | "Yuumi" | "Zilean"
        | "Zyra" => "Sup",

        _ => "Mid",
    }
}

/// Bottom action bar — shows "APPLY RUNES & BUILD" when a build is ready,
/// or "PICK A CHAMPION" to navigate to the champion picker otherwise.
fn champ_select_bar(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    main_w:    u32,
    bar_h:     u32,
) {
    let wrapper = NativeRenderer::element("div");
    NativeRenderer::set_attr(&wrapper, "data-layout",    "column");
    NativeRenderer::set_attr(&wrapper, "data-w",         &main_w.to_string());
    NativeRenderer::set_attr(&wrapper, "data-height",    &bar_h.to_string());
    NativeRenderer::set_attr(&wrapper, "data-fill",   "transparent");
    NativeRenderer::append(parent, &wrapper);

    let btn_h = bar_h.saturating_sub(40);
    let btn_w = (main_w * 2 / 5).max(280);
    let row   = hrow(btn_h);
    NativeRenderer::append(&wrapper, &row);

    let spc = NativeRenderer::element("div");
    NativeRenderer::set_attr(&spc, "data-flex",   "0.5");
    NativeRenderer::set_attr(&spc, "data-height", &btn_h.to_string());
    NativeRenderer::append(&row, &spc);

    // Resolve the active rune page for the apply action.
    let active_page = state.current_build.as_ref().map(|build| {
        let pages: Vec<&RuneRecommendation> = std::iter::once(&build.runes)
            .chain(build.alt_runes.iter())
            .take(3)
            .collect();
        let idx = state.selected_rune_page.min(pages.len().saturating_sub(1));
        pages[idx].clone()
    });

    let applying = matches!(state.rune_status, RuneStatus::Applying);

    if let Some(page) = active_page {
        // Build is ready — show the Apply button.
        let (status_text, status_color) = rune_status_line(&state.rune_status, theme);

        let btn = NativeRenderer::element("div");
        NativeRenderer::set_attr(&btn, "data-layout", "column");
        NativeRenderer::set_attr(&btn, "data-w",      &btn_w.to_string());
        NativeRenderer::set_attr(&btn, "data-h",      &btn_h.to_string());
        NativeRenderer::set_attr(&btn, "data-fill",         if applying { theme.muted } else { theme.bg });
        NativeRenderer::set_attr(&btn, "data-align",         "center");
        NativeRenderer::set_attr(&btn, "data-border-top",    &format!("{}:2", theme.accent));
        NativeRenderer::set_attr(&btn, "data-border-bottom", &format!("{}:2", theme.accent));
        NativeRenderer::set_attr(&btn, "data-border-left",   &format!("{}:2", theme.accent));
        NativeRenderer::set_attr(&btn, "data-border-right",  &format!("{}:2", theme.accent));
        if !applying {
            NativeRenderer::set_attr(&btn, "data-hover-fill", theme.surface);
        }

        let main_lbl = NativeRenderer::text(if applying { "APPLYING…" } else { "APPLY RUNES & BUILD" });
        NativeRenderer::set_attr(&main_lbl, "data-color",       "#ffffff");
        NativeRenderer::set_attr(&main_lbl, "data-text-size",   "21");
        NativeRenderer::set_attr(&main_lbl, "data-text-weight", "bold");
        NativeRenderer::set_attr(&main_lbl, "data-align",       "center");
        NativeRenderer::set_attr(&main_lbl, "data-h",           "32");
        NativeRenderer::set_attr(&main_lbl, "data-pad",         "10 0 0 0");
        NativeRenderer::append(&btn, &main_lbl);

        let sub_text = if !status_text.is_empty() { status_text } else { format!("— {} —", page.name) };
        let sub_lbl = NativeRenderer::text(&sub_text);
        NativeRenderer::set_attr(&sub_lbl, "data-color",     if matches!(state.rune_status, RuneStatus::Idle | RuneStatus::Applying) { theme.accent2 } else { status_color });
        NativeRenderer::set_attr(&sub_lbl, "data-text-size", "15");
        NativeRenderer::set_attr(&sub_lbl, "data-align",     "center");
        NativeRenderer::set_attr(&sub_lbl, "data-h",         "18");
        NativeRenderer::append(&btn, &sub_lbl);

        if !applying {
            NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
                if let Ok(mut s) = state_arc.lock() {
                    s.rune_status = RuneStatus::Applying;
                    let _ = s.rune_tx.try_send(RuneCmd::Apply(page.clone()));
                }
            }));
        }
        NativeRenderer::append(&row, &btn);
    } else {
        // No build loaded — navigate to champion picker.
        let btn = NativeRenderer::element("div");
        NativeRenderer::set_attr(&btn, "data-layout",     "row");
        NativeRenderer::set_attr(&btn, "data-w",          &btn_w.to_string());
        NativeRenderer::set_attr(&btn, "data-height",     &btn_h.to_string());
        NativeRenderer::set_attr(&btn, "data-fill",       theme.surface_hi);
        NativeRenderer::set_attr(&btn, "data-hover-fill", "#252535");
        NativeRenderer::set_attr(&btn, "data-border-left",&format!("{}:3", theme.accent));
        NativeRenderer::set_attr(&btn, "data-pad",        "0 12 0 18");
        NativeRenderer::append(&row, &btn);

        let lbl = NativeRenderer::text("PICK A CHAMPION");
        NativeRenderer::set_attr(&lbl, "data-color",       theme.text);
        NativeRenderer::set_attr(&lbl, "data-text-size",   "17");
        NativeRenderer::set_attr(&lbl, "data-text-weight", "bold");
        NativeRenderer::set_attr(&lbl, "data-flex",        "1.0");
        NativeRenderer::set_attr(&lbl, "data-h",           &btn_h.to_string());
        NativeRenderer::set_attr(&lbl, "data-align",       "center");
        NativeRenderer::append(&btn, &lbl);

        NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
            if let Ok(mut s) = state_arc.lock() { s.active_tab = Tab::Champions; }
        }));
        NativeRenderer::append(&row, &btn);
    }

    let spc2 = NativeRenderer::element("div");
    NativeRenderer::set_attr(&spc2, "data-flex",   "0.5");
    NativeRenderer::set_attr(&spc2, "data-height", &btn_h.to_string());
    NativeRenderer::append(&row, &spc2);
}

/// Full-width champion picker: search + role filters + scrollable grid.
fn champion_tab(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    main_w:    u32,
    body_h:    u32,
) {
    const CELL: u32  = 140;
    let cols         = ((main_w / CELL) as usize).max(2);
    const SEARCH_H: u32   = 40;
    const FILTER_H: u32   = 44;
    const CHIP_W:   u32   = 54;

    let selected = state.selected_champion.as_deref().unwrap_or("");

    // Search box
    {
        let search_row = hrow(SEARCH_H);
        NativeRenderer::set_attr(&search_row, "data-fill",          "#1c1c2c");
        NativeRenderer::set_attr(&search_row, "data-border-bottom", &format!("{}:1", theme.accent));
        NativeRenderer::set_attr(&search_row, "data-pad",           "0 0 0 10");
        let icon = NativeRenderer::text("/ ");
        NativeRenderer::set_attr(&icon, "data-color",     theme.accent);
        NativeRenderer::set_attr(&icon, "data-text-size", "16");
        NativeRenderer::set_attr(&icon, "data-h",         &SEARCH_H.to_string());
        NativeRenderer::append(&search_row, &icon);
        let q = if state.search_query.is_empty() {
            "search\u{2026}".to_string()
        } else {
            format!("{}|", state.search_query)
        };
        let query = NativeRenderer::text(&q);
        NativeRenderer::set_attr(&query, "data-color",
            if state.search_query.is_empty() { theme.muted } else { theme.text });
        NativeRenderer::set_attr(&query, "data-text-size", "14");
        NativeRenderer::set_attr(&query, "data-h",         &SEARCH_H.to_string());
        NativeRenderer::append(&search_row, &query);
        NativeRenderer::append(parent, &search_row);
    }

    // Role filter chips
    {
        let filter_row = hrow(FILTER_H);
        NativeRenderer::set_attr(&filter_row, "data-pad", "0 0 0 10");
        for role in [RoleFilter::All, RoleFilter::Top, RoleFilter::Jng,
                     RoleFilter::Mid, RoleFilter::Bot, RoleFilter::Sup] {
            let is_active = state.champ_role_filter == role;
            let chip = NativeRenderer::element("div");
            NativeRenderer::set_attr(&chip, "data-w",          &CHIP_W.to_string());
            NativeRenderer::set_attr(&chip, "data-h",          &FILTER_H.to_string());
            NativeRenderer::set_attr(&chip, "data-fill",       if is_active { theme.accent } else { theme.surface_hi });
            NativeRenderer::set_attr(&chip, "data-hover-fill", if is_active { "#A01820" } else { "#252535" });
            if is_active {
                NativeRenderer::set_attr(&chip, "data-border-bottom", &format!("{}:2", theme.accent));
            }
            let t = NativeRenderer::text(role.label());
            NativeRenderer::set_attr(&t, "data-color",       if is_active { "#ffffff" } else { theme.text });
            NativeRenderer::set_attr(&t, "data-text-size",   "13");
            NativeRenderer::set_attr(&t, "data-text-weight", "bold");
            NativeRenderer::set_attr(&t, "data-h",           &FILTER_H.to_string());
            NativeRenderer::set_attr(&t, "data-align",       "center");
            NativeRenderer::append(&chip, &t);
            let sa = state_arc.clone();
            NativeRenderer::on_event(&chip, "click", Box::new(move |_: BrickEvent| {
                if let Ok(mut s) = sa.lock() { s.champ_role_filter = role; }
            }));
            NativeRenderer::append(&filter_row, &chip);
        }
        NativeRenderer::append(parent, &filter_row);
    }

    // Champion grid (full-width, scrollable)
    let grid_h    = body_h.saturating_sub(SEARCH_H + 1 + FILTER_H);
    let side_pad  = (main_w.saturating_sub(cols as u32 * CELL)) / 2;

    let grid_scroll = NativeRenderer::element("div");
    NativeRenderer::set_attr(&grid_scroll, "data-scroll-y",  "true");
    NativeRenderer::set_attr(&grid_scroll, "data-scroll-id", "champion-grid");
    NativeRenderer::set_attr(&grid_scroll, "data-height",    &grid_h.to_string());
    NativeRenderer::append(parent, &grid_scroll);

    let search = state.search_query.as_str();
    let role_filter = match state.champ_role_filter {
        RoleFilter::All => None,
        RoleFilter::Top => Some("Top"),
        RoleFilter::Jng => Some("Jng"),
        RoleFilter::Mid => Some("Mid"),
        RoleFilter::Bot => Some("Bot"),
        RoleFilter::Sup => Some("Sup"),
    };
    let visible: Vec<&str> = ALL_CHAMPIONS.iter()
        .copied()
        .filter(|&name| {
            (search.is_empty() || name.to_lowercase().starts_with(search)) &&
            role_filter.map_or(true, |r| champion_primary_role(name) == r)
        })
        .collect();
    for chunk in visible.chunks(cols) {
        let row = hrow(CELL);
        NativeRenderer::set_attr(&row, "data-pad", &format!("0 0 0 {side_pad}"));
        for &name in chunk {
            NativeRenderer::append(
                &row,
                &champion_button(name, name == selected, state_arc.clone(), theme, CELL),
            );
        }
        NativeRenderer::append(&grid_scroll, &row);
    }
}

fn rune_tab(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    main_w:    u32,
    body_h:    u32,
) {
    // Splash: data-overlay so it takes no space in the flow but paints first
    // (behind the bar and content).  Positioned at the top-right quadrant.
    if let Some(name) = &state.selected_champion {
        let splash_w = main_w / 2;
        let splash_h = body_h / 2;
        let splash = NativeRenderer::element("div");
        NativeRenderer::set_attr(&splash, "data-overlay", "true");
        NativeRenderer::set_attr(&splash, "data-x", &(SIDEBAR_W + main_w / 2).to_string());
        NativeRenderer::set_attr(&splash, "data-y", &TITLEBAR_H.to_string());
        NativeRenderer::set_attr(&splash, "data-w", &splash_w.to_string());
        NativeRenderer::set_attr(&splash, "data-h", &splash_h.to_string());
        NativeRenderer::set_attr(&splash, "data-fill",        theme.bg);
        NativeRenderer::set_attr(&splash, "data-image",       &format!("assets/champion_splashes/{name}_0.jpg"));
        NativeRenderer::set_attr(&splash, "data-image-alpha", "0.45");
        NativeRenderer::set_attr(&splash, "data-fade-to",     theme.bg);
        NativeRenderer::set_attr(&splash, "data-fade-bottom", &(splash_h * 2 / 3).to_string());
        NativeRenderer::set_attr(&splash, "data-fade-left",   "100");
        NativeRenderer::append(parent, &splash);
    }

    // Bar and content flow normally from the top, painting on top of the overlay.
    let bar_h = if state.selected_champion.is_some() {
        rune_champ_header(state, state_arc.clone(), theme, parent, main_w)
    } else {
        0
    };
    let content_h = body_h.saturating_sub(bar_h);

    match (&state.current_build, &state.selected_champion) {
        (Some(build), Some(_)) => build_preview(build, state, state_arc, theme, parent, main_w, content_h),
        _ => {
            spacer(40, parent);
            let hint = NativeRenderer::text("\u{2190}  Select a champion first");
            NativeRenderer::set_attr(&hint, "data-color",     theme.muted);
            NativeRenderer::set_attr(&hint, "data-text-size", "16");
            NativeRenderer::set_attr(&hint, "data-pad",       "0 0 0 32");
            NativeRenderer::set_attr(&hint, "data-h",         "24");
            NativeRenderer::append(parent, &hint);
        }
    }
}

fn champion_button(
    name:      &str,
    selected:  bool,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    size:      u32,
) -> NativeNode {
    // data-image blits over data-fill, so btn must have no children: hit_test
    // returns the deepest node and hover/fill only applies to the hit target.
    let btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&btn, "data-w",      &size.to_string());
    NativeRenderer::set_attr(&btn, "data-height", &size.to_string());
    NativeRenderer::set_attr(&btn, "data-image",  &format!("assets/champion_icons/{name}.png"));

    if selected {
        NativeRenderer::set_attr(&btn, "data-fill",          "#2a0a12");
        NativeRenderer::set_attr(&btn, "data-border-bottom", &format!("{}:3", theme.accent));
        NativeRenderer::set_attr(&btn, "data-hover-fill",    "#3a1020");
    } else {
        NativeRenderer::set_attr(&btn, "data-fill",       "transparent");
        NativeRenderer::set_attr(&btn, "data-hover-fill", theme.surface_hi);
    }

    let name_owned = name.to_string();
    NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = state_arc.lock() {
            let build = s.recommender.build_for(&name_owned);
            s.selected_champion  = Some(name_owned.clone());
            s.current_build      = Some(build);
            s.rune_status        = RuneStatus::Idle;
            s.selected_rune_page = 0;
            s.patch_changes     = vec![];
            s.patch_loading     = true;
            let _ = s.fetch_tx.try_send(FetchCmd::All(name_owned.clone()));
            if s.active_tab == Tab::Champions {
                s.active_tab = Tab::Runes;
            }
        }
    }));

    btn
}

/// Empty-state right column shown when no champion is selected.
fn build_placeholder(theme: &Theme, parent: &NativeNode) {
    spacer(120, parent);

    // Card container — fixed width, left-padded to feel anchored
    let card = NativeRenderer::element("div");
    NativeRenderer::set_attr(&card, "data-layout",      "column");
    NativeRenderer::set_attr(&card, "data-w",           "380");
    NativeRenderer::set_attr(&card, "data-fill",        theme.surface);
    NativeRenderer::set_attr(&card, "data-border-left", &format!("{}:3", theme.accent));
    NativeRenderer::set_attr(&card, "data-pad",         "24 32 28 32");
    NativeRenderer::set_attr(&card, "data-align",       "center");
    NativeRenderer::append(parent, &card);

    // Heading: left-align via a row + flex spacer
    {
        let hdr_row = hrow(28);
        let hdr = NativeRenderer::text("HOW IT WORKS");
        NativeRenderer::set_attr(&hdr, "data-color",       theme.accent);
        NativeRenderer::set_attr(&hdr, "data-text-size",   "13");
        NativeRenderer::set_attr(&hdr, "data-text-weight", "bold");
        NativeRenderer::set_attr(&hdr, "data-h",           "28");
        NativeRenderer::append(&hdr_row, &hdr);
        NativeRenderer::append(&card, &hdr_row);
    }
    spacer(16, &card);

    for (n, label, sub) in [
        ("1", "Search or browse a champion",  "Type a name to jump straight there"),
        ("2", "Select a portrait",            "Click to load the recommended build"),
        ("3", "Apply runes",                  "One click writes the page via LCU"),
    ] {
        let row = hrow(44);
        // Step number badge
        let badge = NativeRenderer::element("div");
        NativeRenderer::set_attr(&badge, "data-w",          "28");
        NativeRenderer::set_attr(&badge, "data-height",     "28");
        NativeRenderer::set_attr(&badge, "data-fill",       theme.accent);
        NativeRenderer::set_attr(&badge, "data-pad",        "4 0 0 0");
        let n_lbl = NativeRenderer::text(n);
        NativeRenderer::set_attr(&n_lbl, "data-color",       "#ffffff");
        NativeRenderer::set_attr(&n_lbl, "data-text-size",   "14");
        NativeRenderer::set_attr(&n_lbl, "data-text-weight", "bold");
        NativeRenderer::set_attr(&n_lbl, "data-h",           "20");
        NativeRenderer::set_attr(&n_lbl, "data-align",       "center");
        NativeRenderer::append(&badge, &n_lbl);
        NativeRenderer::append(&row, &badge);

        // Text column
        let gap = NativeRenderer::element("div");
        NativeRenderer::set_attr(&gap, "data-w", "14");
        NativeRenderer::append(&row, &gap);

        let txt_col = NativeRenderer::element("div");
        NativeRenderer::set_attr(&txt_col, "data-layout", "column");
        NativeRenderer::set_attr(&txt_col, "data-flex",   "1.0");
        NativeRenderer::set_attr(&txt_col, "data-h",      "44");
        NativeRenderer::set_attr(&txt_col, "data-pad",    "4 0 0 0");

        let main_lbl = NativeRenderer::text(label);
        NativeRenderer::set_attr(&main_lbl, "data-color",       theme.text);
        NativeRenderer::set_attr(&main_lbl, "data-text-size",   "15");
        NativeRenderer::set_attr(&main_lbl, "data-text-weight", "bold");
        NativeRenderer::set_attr(&main_lbl, "data-h",           "22");
        NativeRenderer::append(&txt_col, &main_lbl);

        let sub_lbl = NativeRenderer::text(sub);
        NativeRenderer::set_attr(&sub_lbl, "data-color",     theme.muted);
        NativeRenderer::set_attr(&sub_lbl, "data-text-size", "14");
        NativeRenderer::set_attr(&sub_lbl, "data-h",         "18");
        NativeRenderer::append(&txt_col, &sub_lbl);

        NativeRenderer::append(&row, &txt_col);
        NativeRenderer::append(&card, &row);
        spacer(6, &card);
    }
}

/// Return the canonical hex color for a rune path style id.
fn rune_path_color(style_id: u32) -> &'static str {
    match style_id {
        8000 => "#C89B3C", // Precision — gold
        8100 => "#C84B4B", // Domination — crimson
        8200 => "#4A6BB5", // Sorcery — sapphire
        8300 => "#49C2C2", // Inspiration — teal
        8400 => "#4CB85C", // Resolve — emerald
        _    => "#A0A0A0", // unknown
    }
}

/// Linearly interpolate two hex-RGB colors (format `"#RRGGBB"`).
/// `t` = 0.0 → `a`, `t` = 1.0 → `b`.
fn lerp_hex(a: &str, b: &str, t: f32) -> String {
    let parse = |s: &str, lo: usize, hi: usize| -> f32 {
        u8::from_str_radix(s.get(lo..hi).unwrap_or("00"), 16).unwrap_or(0) as f32
    };
    let r = (parse(a, 1, 3) + (parse(b, 1, 3) - parse(a, 1, 3)) * t) as u8;
    let g = (parse(a, 3, 5) + (parse(b, 3, 5) - parse(a, 3, 5)) * t) as u8;
    let bv = (parse(a, 5, 7) + (parse(b, 5, 7) - parse(a, 5, 7)) * t) as u8;
    format!("#{r:02X}{g:02X}{bv:02X}")
}

/// Vertical rune tree column with a centered progression spine and pulsing
/// path-colored circle outlines around each rune icon.
fn rune_tree_column(
    header:     &str,
    is_primary: bool,
    style_id:   u32,
    perk_ids:   &[u32],
    theme:      &Theme,
    col_w:      u32,
) -> NativeNode {
    // Pulse: smooth 1.5-second cycle, 0.0 (dim) → 1.0 (bright).
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f32();
    let pulse     = ((secs / 1.5 * std::f32::consts::TAU).sin() + 1.0) * 0.5;
    let path_col  = rune_path_color(style_id);
    let dim_col   = lerp_hex("#282828", path_col, 0.30);
    let spine_col = lerp_hex(&dim_col, path_col, pulse);

    // Spine zone: wide enough to center the largest icon in this column.
    // Both keystone (160) and secondary (80) icons share the same center so
    // the spine runs straight through all slots.
    let max_icon: u32 = if is_primary { 160 } else { 80 };
    let spine_zone_w  = max_icon + 20; // 10px margin each side
    let spine_cx      = spine_zone_w / 2; // center of zone in local coords

    let col = NativeRenderer::element("div");
    NativeRenderer::set_attr(&col, "data-layout", "column");
    NativeRenderer::set_attr(&col, "data-w",      &col_w.to_string());
    NativeRenderer::set_attr(&col, "data-fill",   theme.surface);

    // Path header bar
    let hdr = hrow(40);
    NativeRenderer::set_attr(&hdr, "data-fill",        theme.surface_hi);
    NativeRenderer::set_attr(&hdr, "data-border-left", &format!("{path_col}:3"));
    NativeRenderer::set_attr(&hdr, "data-pad",         "0 0 0 14");
    let t = NativeRenderer::text(header);
    NativeRenderer::set_attr(&t, "data-color",
        if is_primary { theme.accent } else { theme.accent2 });
    NativeRenderer::set_attr(&t, "data-text-size",   "14");
    NativeRenderer::set_attr(&t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&t, "data-h",           "40");
    NativeRenderer::append(&hdr, &t);
    NativeRenderer::append(&col, &hdr);

    if perk_ids.is_empty() { return col; }

    let rune_icon  = |id: u32| format!("assets/rune_icons/{id}.png");
    let perk_label = |id: u32| -> String {
        let n = perk_name(id);
        if n.is_empty() { format!("#{id}") } else { n.to_string() }
    };

    for (i, &id) in perk_ids.iter().enumerate() {
        let is_keystone = is_primary && i == 0;
        let icon_sz: u32 = if is_keystone { 160 } else { 80 };
        let v_pad:   u32 = if is_keystone { 24 } else { 20 };
        let row_h    = v_pad + icon_sz + v_pad;

        let slot = hrow(row_h);
        NativeRenderer::set_attr(&slot, "data-fill", theme.surface);

        // ── Spine zone: column so the icon sits at v_pad from the top, making
        //    its vertical center equal to slot.y + v_pad + icon_sz/2 — the same
        //    anchor the text column uses.
        let zone = NativeRenderer::element("div");
        NativeRenderer::set_attr(&zone, "data-layout", "column");
        NativeRenderer::set_attr(&zone, "data-w",      &spine_zone_w.to_string());
        NativeRenderer::set_attr(&zone, "data-height", &row_h.to_string());

        spacer(v_pad, &zone); // pushes icon down by v_pad

        // Inner row: horizontal centering of the icon within the spine zone.
        let icon_row = NativeRenderer::element("div");
        NativeRenderer::set_attr(&icon_row, "data-layout", "row");
        let icon_left = (spine_zone_w - icon_sz) / 2;
        let zone_spc = NativeRenderer::element("div");
        NativeRenderer::set_attr(&zone_spc, "data-w", &icon_left.to_string());
        NativeRenderer::append(&icon_row, &zone_spc);

        // Rune icon with pulsing circle border.
        let icon_border_w = if is_keystone { 3u32 } else { 2 };
        let img = NativeRenderer::element("div");
        NativeRenderer::set_attr(&img, "data-w",             &icon_sz.to_string());
        NativeRenderer::set_attr(&img, "data-height",        &icon_sz.to_string());
        NativeRenderer::set_attr(&img, "data-image",         &rune_icon(id));
        NativeRenderer::set_attr(&img, "data-circle-border", &format!("{spine_col}:{icon_border_w}"));
        NativeRenderer::append(&icon_row, &img);
        NativeRenderer::append(&zone, &icon_row);
        NativeRenderer::append(&slot, &zone);

        // ── Text column ──────────────────────────────────────────────────────
        let gap_w = if is_keystone { 20u32 } else { 16 };
        let gap = NativeRenderer::element("div");
        NativeRenderer::set_attr(&gap, "data-w", &gap_w.to_string());
        NativeRenderer::append(&slot, &gap);

        let font_sz: u32 = if is_keystone { 24 } else { 18 };
        let desc_sz: u32 = if is_keystone { 15 } else { 13 };
        let desc = perk_desc(id);
        let name_h:  u32 = font_sz + 4;
        let avail_text_w = col_w.saturating_sub(spine_zone_w + gap_w + 16);
        let desc_h:  u32 = if desc.is_empty() { 0 } else {
            NativeRenderer::measure_text(desc, avail_text_w, desc_sz as f32)
        };
        let desc_gap: u32 = if desc.is_empty() { 0 } else { 6 };
        let block_h  = name_h + desc_gap + desc_h;
        let top_pad  = icon_sz.saturating_sub(block_h) / 2 + v_pad;

        let text_col = NativeRenderer::element("div");
        NativeRenderer::set_attr(&text_col, "data-layout", "column");
        NativeRenderer::set_attr(&text_col, "data-flex",   "1.0");
        NativeRenderer::set_attr(&text_col, "data-pad",    &format!("{top_pad} 16 0 0"));

        let lbl = NativeRenderer::text(&perk_label(id));
        NativeRenderer::set_attr(&lbl, "data-color",
            if is_keystone { theme.text } else { theme.accent2 });
        NativeRenderer::set_attr(&lbl, "data-text-size", &font_sz.to_string());
        if is_keystone {
            NativeRenderer::set_attr(&lbl, "data-text-weight", "bold");
        }
        NativeRenderer::append(&text_col, &lbl);

        if !desc.is_empty() {
            let d = NativeRenderer::text(desc);
            NativeRenderer::set_attr(&d, "data-color",     theme.muted);
            NativeRenderer::set_attr(&d, "data-text-size", &desc_sz.to_string());
            NativeRenderer::set_attr(&d, "data-pad",       &format!("{desc_gap} 0 0 0"));
            NativeRenderer::append(&text_col, &d);
        }

        NativeRenderer::append(&slot, &text_col);
        NativeRenderer::append(&col, &slot);

        // Spine connector after this rune (before the next rune row).
        if i < perk_ids.len() - 1 {
            let conn = hrow(16);
            let spc = NativeRenderer::element("div");
            NativeRenderer::set_attr(&spc, "data-w", &(spine_cx - 1).to_string());
            NativeRenderer::append(&conn, &spc);
            let line = NativeRenderer::element("div");
            NativeRenderer::set_attr(&line, "data-w",    "2");
            NativeRenderer::set_attr(&line, "data-fill", &spine_col);
            NativeRenderer::append(&conn, &line);
            NativeRenderer::append(&col, &conn);
        }
    }

    col
}

fn build_preview(
    build:     &BuildRecommendation,
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    main_w:    u32,
    tab_h:     u32,
) {
    // Collect up to 3 rune pages; primary first, then alternates.
    let all_pages: Vec<&RuneRecommendation> = std::iter::once(&build.runes)
        .chain(build.alt_runes.iter())
        .take(3)
        .collect();
    // page_idx may exceed all_pages.len() — the rune section uses get() to guard.
    let page_idx = state.selected_rune_page.min(2);

    // Scrollable container — fills the remaining tab height.
    let scroll = NativeRenderer::element("div");
    NativeRenderer::set_attr(&scroll, "data-scroll-y",  "true");
    NativeRenderer::set_attr(&scroll, "data-scroll-id", "build-scroll");
    NativeRenderer::set_attr(&scroll, "data-height",    &tab_h.to_string());
    NativeRenderer::append(parent, &scroll);

    // Side-by-side layout: rune card on the left, items card on the right.
    let side_pad = 12u32;
    let gap      = 8u32;
    let items_w  = 360u32;
    let items_fw = items_w + 4;
    let rune_w   = main_w.saturating_sub(side_pad + gap + items_fw + 4).min(1060).max(500);
    let rune_fw  = rune_w + 4;

    spacer(12, &scroll);
    let center_row = NativeRenderer::element("div");
    NativeRenderer::set_attr(&center_row, "data-layout", "row");
    NativeRenderer::append(&scroll, &center_row);

    let lspc = NativeRenderer::element("div");
    NativeRenderer::set_attr(&lspc, "data-w", &side_pad.to_string());
    NativeRenderer::set_attr(&lspc, "data-h", "1");
    NativeRenderer::append(&center_row, &lspc);

    let card_frame = NativeRenderer::element("div");
    NativeRenderer::set_attr(&card_frame, "data-layout", "column");
    NativeRenderer::set_attr(&card_frame, "data-w",      &rune_fw.to_string());
    NativeRenderer::set_attr(&card_frame, "data-fill",   theme.accent);
    NativeRenderer::set_attr(&card_frame, "data-pad",    "2 2 2 2");
    NativeRenderer::append(&center_row, &card_frame);

    let card = NativeRenderer::element("div");
    NativeRenderer::set_attr(&card, "data-layout", "column");
    NativeRenderer::set_attr(&card, "data-w",      &rune_w.to_string());
    NativeRenderer::set_attr(&card, "data-fill",   theme.surface);
    NativeRenderer::append(&card_frame, &card);

    // ── Rune page tabs — 3 equal-width slots spanning the full card width ────────
    {
        const TAB_H: u32 = 100;
        let tabs_row = hrow(TAB_H);
        NativeRenderer::set_attr(&tabs_row, "data-w",             &rune_w.to_string());
        NativeRenderer::set_attr(&tabs_row, "data-fill",          theme.bg);
        NativeRenderer::set_attr(&tabs_row, "data-border-bottom", &format!("{}:2", theme.accent));

        for i in 0..3usize {
            let has_page  = i < all_pages.len();
            let is_active = i == page_idx;

            let wr_label = if has_page {
                let p = all_pages[i];
                match (p.win_rate, p.pick_rate) {
                    (Some(wr), Some(pr)) => format!("{wr:.1}% WR  ·  {pr:.1}% PR"),
                    (Some(wr), None)     => format!("{wr:.1}% WR"),
                    (None, Some(pr))     => format!("{pr:.1}% PR"),
                    (None, None)         => "\u{2014}".to_string(),
                }
            } else {
                "No data".to_string()
            };

            // Last tab absorbs rounding so all three sum exactly to card_w.
            let tab_w = if i < 2 { rune_w / 3 } else { rune_w - (rune_w / 3) * 2 };

            // Tab: row layout — keystone icon left, text column right.
            let tab = NativeRenderer::element("div");
            NativeRenderer::set_attr(&tab, "data-layout", "row");
            NativeRenderer::set_attr(&tab, "data-w",      &tab_w.to_string());
            NativeRenderer::set_attr(&tab, "data-h",      &TAB_H.to_string());
            NativeRenderer::set_attr(&tab, "data-fill",
                if is_active { theme.surface_hi } else { theme.bg });
            if is_active {
                NativeRenderer::set_attr(&tab, "data-border-bottom", &format!("{}:4", theme.accent));
            } else {
                NativeRenderer::set_attr(&tab, "data-hover-fill", theme.surface);
            }
            if i < 2 {
                NativeRenderer::set_attr(&tab, "data-border-right", &format!("{}:1", theme.divider));
            }

            // Keystone icon — fixed-width wrapper, vertically centred in the tab.
            const KS_ICON_SZ:   u32 = 56;
            const KS_ICON_WRAP: u32 = 88; // 16px gap + 56px icon + 16px gap
            let ks_v_pad = (TAB_H - KS_ICON_SZ) / 2;
            let icon_wrap = NativeRenderer::element("div");
            NativeRenderer::set_attr(&icon_wrap, "data-layout", "row");
            NativeRenderer::set_attr(&icon_wrap, "data-w",      &KS_ICON_WRAP.to_string());
            NativeRenderer::set_attr(&icon_wrap, "data-height", &TAB_H.to_string());
            NativeRenderer::set_attr(&icon_wrap, "data-pad",
                &format!("{ks_v_pad} 16 {ks_v_pad} 16"));
            if has_page {
                let ks_id = all_pages[i].selected_perk_ids.first().copied().unwrap_or(0);
                let ks_icon = NativeRenderer::element("div");
                NativeRenderer::set_attr(&ks_icon, "data-w",      &KS_ICON_SZ.to_string());
                NativeRenderer::set_attr(&ks_icon, "data-height", &KS_ICON_SZ.to_string());
                NativeRenderer::set_attr(&ks_icon, "data-image",
                    &format!("assets/rune_icons/{ks_id}.png"));
                NativeRenderer::append(&icon_wrap, &ks_icon);
            }
            NativeRenderer::append(&tab, &icon_wrap);

            // Text column — path name + win rate, vertically centred.
            let text_col = NativeRenderer::element("div");
            NativeRenderer::set_attr(&text_col, "data-layout", "column");
            NativeRenderer::set_attr(&text_col, "data-flex",   "1.0");
            NativeRenderer::set_attr(&text_col, "data-h",      &TAB_H.to_string());
            NativeRenderer::set_attr(&text_col, "data-pad",    "26 0 0 0");

            let path_str = if has_page {
                let ks_id = all_pages[i].selected_perk_ids.first().copied().unwrap_or(0);
                let n = perk_name(ks_id);
                if n.is_empty() { rune_path_name(all_pages[i].primary_style_id).to_string() }
                else { n.to_string() }
            } else {
                "\u{2014}".to_string()
            };
            let name_t = NativeRenderer::text(&path_str);
            NativeRenderer::set_attr(&name_t, "data-color",
                if is_active     { theme.accent }
                else if has_page { theme.text }
                else             { theme.muted });
            NativeRenderer::set_attr(&name_t, "data-text-size",   "20");
            NativeRenderer::set_attr(&name_t, "data-text-weight", if is_active { "bold" } else { "normal" });
            NativeRenderer::append(&text_col, &name_t);

            let wr_t = NativeRenderer::text(&wr_label);
            NativeRenderer::set_attr(&wr_t, "data-color",     if has_page { theme.accent2 } else { theme.divider });
            NativeRenderer::set_attr(&wr_t, "data-text-size", "14");
            NativeRenderer::set_attr(&wr_t, "data-pad",       "6 0 0 0");
            NativeRenderer::append(&text_col, &wr_t);

            NativeRenderer::append(&tab, &text_col);

            let sa = state_arc.clone();
            NativeRenderer::on_event(&tab, "click", Box::new(move |_: BrickEvent| {
                if let Ok(mut s) = sa.lock() { s.selected_rune_page = i; }
            }));

            NativeRenderer::append(&tabs_row, &tab);
        }
        NativeRenderer::append(&card, &tabs_row);
        spacer(12, &card);
    }

    // ── Rune Setup ────────────────────────────────────────────────────────────
    if let Some(active) = all_pages.get(page_idx) {
        let perks = &active.selected_perk_ids;

        let primary_w   = (rune_w * 58 / 100).max(200);
        let secondary_w = rune_w.saturating_sub(primary_w + 1);

        let runes_row = NativeRenderer::element("div");
        NativeRenderer::set_attr(&runes_row, "data-layout", "row");
        NativeRenderer::set_attr(&runes_row, "data-w",      &rune_w.to_string());

        NativeRenderer::append(&runes_row, &rune_tree_column(
            &format!("PRIMARY  ·  {}", rune_path_name(active.primary_style_id)),
            true,
            active.primary_style_id,
            &perks[..perks.len().min(4)],
            theme,
            primary_w,
        ));

        let vdiv = NativeRenderer::element("div");
        NativeRenderer::set_attr(&vdiv, "data-w",    "1");
        NativeRenderer::set_attr(&vdiv, "data-fill", theme.divider);
        NativeRenderer::append(&runes_row, &vdiv);

        // Build secondary column, then append shards before adding to the row.
        let secondary_col = rune_tree_column(
            &format!("SECONDARY  ·  {}", rune_path_name(active.sub_style_id)),
            false,
            active.sub_style_id,
            &perks[4.min(perks.len())..6.min(perks.len())],
            theme,
            secondary_w,
        );

        // Shards inside the secondary column — Offense / Flex / Defense
        if perks.len() >= 9 {
            let shr_div = NativeRenderer::element("div");
            NativeRenderer::set_attr(&shr_div, "data-h",    "1");
            NativeRenderer::set_attr(&shr_div, "data-fill", theme.divider);
            NativeRenderer::append(&secondary_col, &shr_div);

            let shdr = hrow(32);
            NativeRenderer::set_attr(&shdr, "data-pad",        "0 0 0 20");
            NativeRenderer::set_attr(&shdr, "data-fill",       theme.surface_hi);
            NativeRenderer::set_attr(&shdr, "data-border-left",&format!("{}:3", theme.accent));
            let shdr_t = NativeRenderer::text("SHARDS");
            NativeRenderer::set_attr(&shdr_t, "data-color",       theme.accent2);
            NativeRenderer::set_attr(&shdr_t, "data-text-size",   "13");
            NativeRenderer::set_attr(&shdr_t, "data-text-weight", "bold");
            NativeRenderer::set_attr(&shdr_t, "data-h",           "32");
            NativeRenderer::append(&shdr, &shdr_t);
            NativeRenderer::append(&secondary_col, &shdr);

            const SHARD_LABELS: [&str; 3] = ["Offense", "Flex", "Defense"];
            const SHARD_ROW_H:  u32 = 52;
            const SHARD_ICON_SZ: u32 = 36;
            let icon_top = (SHARD_ROW_H - SHARD_ICON_SZ) / 2; // 8 — centers 36px in 52px
            let lbl_top  = (SHARD_ROW_H - 15) / 2;            // centers 13px text
            let stat_top = (SHARD_ROW_H - 16) / 2;            // centers 14px text

            for (si, &id) in perks[6..9].iter().enumerate() {
                let shard_row = hrow(SHARD_ROW_H);
                NativeRenderer::set_attr(&shard_row, "data-pad",        "0 0 0 20");
                NativeRenderer::set_attr(&shard_row, "data-border-left",&format!("{}:1", theme.divider));

                // Label — column + spacer so text is vertically centered on the icon
                let lbl_col = NativeRenderer::element("div");
                NativeRenderer::set_attr(&lbl_col, "data-layout", "column");
                NativeRenderer::set_attr(&lbl_col, "data-w",      "80");
                NativeRenderer::set_attr(&lbl_col, "data-height", &SHARD_ROW_H.to_string());
                spacer(lbl_top, &lbl_col);
                let lbl = NativeRenderer::text(SHARD_LABELS[si]);
                NativeRenderer::set_attr(&lbl, "data-color",     theme.muted);
                NativeRenderer::set_attr(&lbl, "data-text-size", "13");
                NativeRenderer::append(&lbl_col, &lbl);
                NativeRenderer::append(&shard_row, &lbl_col);

                // Icon — column + spacer so 36px image is vertically centered in 52px row
                let icon_col = NativeRenderer::element("div");
                NativeRenderer::set_attr(&icon_col, "data-layout", "column");
                NativeRenderer::set_attr(&icon_col, "data-height", &SHARD_ROW_H.to_string());
                spacer(icon_top, &icon_col);
                let icon = NativeRenderer::element("div");
                NativeRenderer::set_attr(&icon, "data-w",      &SHARD_ICON_SZ.to_string());
                NativeRenderer::set_attr(&icon, "data-height", &SHARD_ICON_SZ.to_string());
                NativeRenderer::set_attr(&icon, "data-image",  &format!("assets/rune_icons/{id}.png"));
                NativeRenderer::set_attr(&icon, "data-fill",   theme.surface_hi);
                NativeRenderer::append(&icon_col, &icon);
                NativeRenderer::append(&shard_row, &icon_col);

                // Stat text — column + spacer so text center aligns with icon center
                let stat = perk_name(id);
                if !stat.is_empty() {
                    let g = NativeRenderer::element("div");
                    NativeRenderer::set_attr(&g, "data-w", "10");
                    NativeRenderer::append(&shard_row, &g);
                    let stat_col = NativeRenderer::element("div");
                    NativeRenderer::set_attr(&stat_col, "data-layout", "column");
                    NativeRenderer::set_attr(&stat_col, "data-height", &SHARD_ROW_H.to_string());
                    spacer(stat_top, &stat_col);
                    let stat_lbl = NativeRenderer::text(stat);
                    NativeRenderer::set_attr(&stat_lbl, "data-color",     theme.accent2);
                    NativeRenderer::set_attr(&stat_lbl, "data-text-size", "14");
                    NativeRenderer::append(&stat_col, &stat_lbl);
                    NativeRenderer::append(&shard_row, &stat_col);
                }

                NativeRenderer::append(&secondary_col, &shard_row);
                if si < 2 {
                    let conn = hrow(6);
                    NativeRenderer::set_attr(&conn, "data-border-left", &format!("{}:1", theme.divider));
                    NativeRenderer::append(&secondary_col, &conn);
                }
            }
        }

        NativeRenderer::append(&runes_row, &secondary_col);
        NativeRenderer::append(&card, &runes_row);
    } else {
        // Selected tab has no data yet.
        spacer(64, &card);
        let msg_row = hrow(32);
        NativeRenderer::set_attr(&msg_row, "data-pad", "0 0 0 32");
        let msg = NativeRenderer::text("No alternate rune page available for this champion.");
        NativeRenderer::set_attr(&msg, "data-color",     theme.muted);
        NativeRenderer::set_attr(&msg, "data-text-size", "15");
        NativeRenderer::set_attr(&msg, "data-h",         "32");
        NativeRenderer::append(&msg_row, &msg);
        NativeRenderer::append(&card, &msg_row);
        spacer(64, &card);
    }

    spacer(12, &card);

    // ── Items card: beside the rune card in center_row ─────────────────────
    let mid = NativeRenderer::element("div");
    NativeRenderer::set_attr(&mid, "data-w", &gap.to_string());
    NativeRenderer::set_attr(&mid, "data-h", "1");
    NativeRenderer::append(&center_row, &mid);

    let bframe = NativeRenderer::element("div");
    NativeRenderer::set_attr(&bframe, "data-layout", "column");
    NativeRenderer::set_attr(&bframe, "data-w",      &items_fw.to_string());
    NativeRenderer::set_attr(&bframe, "data-fill",   theme.accent);
    NativeRenderer::set_attr(&bframe, "data-pad",    "2 2 2 2");
    NativeRenderer::append(&center_row, &bframe);

    let bcard = NativeRenderer::element("div");
    NativeRenderer::set_attr(&bcard, "data-layout", "column");
    NativeRenderer::set_attr(&bcard, "data-w",      &items_w.to_string());
    NativeRenderer::set_attr(&bcard, "data-fill",   theme.surface);
    NativeRenderer::append(&bframe, &bcard);

    spacer(12, &bcard);

    // ── Summoner Spells ───────────────────────────────────────────────────────
    if !build.summoner_spells.is_empty() {
        let spell_row = hrow(72);
        NativeRenderer::set_attr(&spell_row, "data-pad",  "0 0 0 20");
        NativeRenderer::set_attr(&spell_row, "data-fill", theme.bg);

        let hdr = NativeRenderer::text("SPELLS  ");
        NativeRenderer::set_attr(&hdr, "data-color",       theme.muted);
        NativeRenderer::set_attr(&hdr, "data-text-size",   "13");
        NativeRenderer::set_attr(&hdr, "data-text-weight", "bold");
        NativeRenderer::set_attr(&hdr, "data-h",           "72");
        NativeRenderer::append(&spell_row, &hdr);

        for spell in &build.summoner_spells {
            let icon = NativeRenderer::element("div");
            NativeRenderer::set_attr(&icon, "data-w",      "54");
            NativeRenderer::set_attr(&icon, "data-height", "54");
            NativeRenderer::set_attr(&icon, "data-image",
                &format!("assets/summoner_icons/{}.png", spell.to_lowercase()));
            NativeRenderer::set_attr(&icon, "data-fill", theme.surface_hi);
            NativeRenderer::append(&spell_row, &icon);
            let g = NativeRenderer::element("div");
            NativeRenderer::set_attr(&g, "data-w", "10");
            NativeRenderer::append(&spell_row, &g);
        }
        NativeRenderer::append(&bcard, &spell_row);
    }

    // ── Items ─────────────────────────────────────────────────────────────────
    let has_slots = !build.items.slots.is_empty();
    let has_items = !build.items.item_ids.is_empty();
    if has_slots || has_items {
        let hr = NativeRenderer::element("div");
        NativeRenderer::set_attr(&hr, "data-h",    "1");
        NativeRenderer::set_attr(&hr, "data-fill", theme.divider);
        NativeRenderer::append(&bcard, &hr);

        let hdr_row = hrow(36);
        NativeRenderer::set_attr(&hdr_row, "data-pad",  "0 0 0 20");
        NativeRenderer::set_attr(&hdr_row, "data-fill", theme.bg);
        let hdr = NativeRenderer::text("ITEMS");
        NativeRenderer::set_attr(&hdr, "data-color",       theme.muted);
        NativeRenderer::set_attr(&hdr, "data-text-size",   "13");
        NativeRenderer::set_attr(&hdr, "data-text-weight", "bold");
        NativeRenderer::set_attr(&hdr, "data-h",           "36");
        NativeRenderer::append(&hdr_row, &hdr);
        NativeRenderer::append(&bcard, &hdr_row);

        if has_slots {
            // Three-column grid: one column per item slot, up to 3 options each.
            // Height: 8(top) + 52(primary) + 4 + 40(alt1) + 4 + 40(alt2) + 8(bot) = 156
            let slots_row = NativeRenderer::element("div");
            NativeRenderer::set_attr(&slots_row, "data-layout", "row");
            NativeRenderer::set_attr(&slots_row, "data-height", "156");
            NativeRenderer::set_attr(&slots_row, "data-fill",   theme.bg);

            for (si, slot) in build.items.slots.iter().take(3).enumerate() {
                if si > 0 {
                    let vdiv = NativeRenderer::element("div");
                    NativeRenderer::set_attr(&vdiv, "data-w",    "1");
                    NativeRenderer::set_attr(&vdiv, "data-fill", theme.divider);
                    NativeRenderer::append(&slots_row, &vdiv);
                }

                let col = NativeRenderer::element("div");
                NativeRenderer::set_attr(&col, "data-layout", "column");
                NativeRenderer::set_attr(&col, "data-flex",   if si < 2 { "0.333" } else { "0.334" });
                NativeRenderer::set_attr(&col, "data-pad",    "0 0 0 12");

                spacer(8, &col);
                for (oi, &id) in slot.iter().take(3).enumerate() {
                    if oi > 0 { spacer(4, &col); }
                    let icon_sz = if oi == 0 { 52u32 } else { 40 };
                    let icon = NativeRenderer::element("div");
                    NativeRenderer::set_attr(&icon, "data-w",      &icon_sz.to_string());
                    NativeRenderer::set_attr(&icon, "data-height", &icon_sz.to_string());
                    NativeRenderer::set_attr(&icon, "data-image",  &format!("assets/item_icons/{id}.png"));
                    NativeRenderer::set_attr(&icon, "data-fill",   theme.surface_hi);
                    NativeRenderer::append(&col, &icon);
                }
                spacer(8, &col);
                NativeRenderer::append(&slots_row, &col);
            }
            NativeRenderer::append(&bcard, &slots_row);
        } else {
            // Fallback: flat row of up to 6 item icons
            let items_row = hrow(64);
            NativeRenderer::set_attr(&items_row, "data-pad",  "8 0 0 20");
            NativeRenderer::set_attr(&items_row, "data-fill", theme.bg);
            for (idx, &id) in build.items.item_ids.iter().take(6).enumerate() {
                if idx > 0 {
                    let sep = NativeRenderer::element("div");
                    NativeRenderer::set_attr(&sep, "data-w",      "8");
                    NativeRenderer::set_attr(&sep, "data-height", "48");
                    NativeRenderer::append(&items_row, &sep);
                }
                let slot = NativeRenderer::element("div");
                NativeRenderer::set_attr(&slot, "data-w",      "48");
                NativeRenderer::set_attr(&slot, "data-height", "48");
                NativeRenderer::set_attr(&slot, "data-image",  &format!("assets/item_icons/{id}.png"));
                NativeRenderer::set_attr(&slot, "data-fill",   theme.surface_hi);
                NativeRenderer::append(&items_row, &slot);
            }
            NativeRenderer::append(&bcard, &items_row);
        }
    }

    // ── Skill Order ───────────────────────────────────────────────────────────
    if !build.skill_order.is_empty() {
        let hr = NativeRenderer::element("div");
        NativeRenderer::set_attr(&hr, "data-h",    "1");
        NativeRenderer::set_attr(&hr, "data-fill", theme.divider);
        NativeRenderer::append(&bcard, &hr);
        skill_order_section(&build.skill_order, theme, &bcard);
    }

    spacer(20, &bcard);
    spacer(20, &scroll);
}

/// Render a skill max-priority row: R (always first) then the ordered abilities.
/// Each ability is a labelled box with its priority number below.
fn skill_order_section(skill_order: &[String], theme: &Theme, parent: &NativeNode) {
    spacer(12, parent);
    let row = hrow(52);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 24");

    let hdr = NativeRenderer::text("SKILL ORDER  ");
    NativeRenderer::set_attr(&hdr, "data-color",       theme.muted);
    NativeRenderer::set_attr(&hdr, "data-text-size",   "11");
    NativeRenderer::set_attr(&hdr, "data-text-weight", "bold");
    NativeRenderer::set_attr(&hdr, "data-h",           "52");
    NativeRenderer::append(&row, &hdr);

    // R first, then the provided max-order abilities
    let abilities: Vec<&str> = std::iter::once("R")
        .chain(skill_order.iter().map(String::as_str))
        .collect();

    for (i, ability) in abilities.iter().enumerate() {
        let is_ult = *ability == "R";
        let priority = i + 1;

        let col = NativeRenderer::element("div");
        NativeRenderer::set_attr(&col, "data-layout", "column");
        NativeRenderer::set_attr(&col, "data-w",      "36");
        NativeRenderer::set_attr(&col, "data-h",      "52");
        NativeRenderer::set_attr(&col, "data-align",  "center");
        NativeRenderer::set_attr(&col, "data-pad",    "4 0 0 0");

        let badge = NativeRenderer::element("div");
        NativeRenderer::set_attr(&badge, "data-w",    "32");
        NativeRenderer::set_attr(&badge, "data-h",    "28");
        NativeRenderer::set_attr(&badge, "data-fill", if is_ult { theme.accent } else { theme.surface_hi });
        NativeRenderer::set_attr(&badge, "data-align", "center");
        let letter = NativeRenderer::text(ability);
        NativeRenderer::set_attr(&letter, "data-color",       "#ffffff");
        NativeRenderer::set_attr(&letter, "data-text-size",   "14");
        NativeRenderer::set_attr(&letter, "data-text-weight", "bold");
        NativeRenderer::set_attr(&letter, "data-h",           "28");
        NativeRenderer::set_attr(&letter, "data-align",       "center");
        NativeRenderer::append(&badge, &letter);
        NativeRenderer::append(&col, &badge);

        let prio = NativeRenderer::text(&priority.to_string());
        NativeRenderer::set_attr(&prio, "data-color",     theme.muted);
        NativeRenderer::set_attr(&prio, "data-text-size", "11");
        NativeRenderer::set_attr(&prio, "data-h",         "16");
        NativeRenderer::set_attr(&prio, "data-align",     "center");
        NativeRenderer::append(&col, &prio);

        NativeRenderer::append(&row, &col);

        if i < abilities.len() - 1 {
            let arrow = NativeRenderer::text("›");
            NativeRenderer::set_attr(&arrow, "data-color",     theme.muted);
            NativeRenderer::set_attr(&arrow, "data-text-size", "18");
            NativeRenderer::set_attr(&arrow, "data-h",         "52");
            NativeRenderer::set_attr(&arrow, "data-pad",       "0 4 0 4");
            NativeRenderer::append(&row, &arrow);
        }
    }

    NativeRenderer::append(parent, &row);
}


fn rune_status_line<'a>(status: &RuneStatus, theme: &'a Theme) -> (String, &'a str) {
    match status {
        RuneStatus::Idle          => (String::new(), theme.muted),
        RuneStatus::Applying      => ("Applying rune page via LCU…".to_string(), theme.accent2),
        RuneStatus::Applied(msg)  => (format!("✓ {msg}"), theme.ok),
        RuneStatus::Error(msg)    => (format!("✗ {msg}"), theme.err),
    }
}

// ── Patch notes tab ───────────────────────────────────────────────────────────

/// `[-] Np [+]` patch-window control. Appended into a row node.
fn patch_window_ctrl(
    lookback:  u32,
    champion:  Option<String>,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
) {
    let ctrl = NativeRenderer::element("div");
    NativeRenderer::set_attr(&ctrl, "data-layout", "row");
    NativeRenderer::set_attr(&ctrl, "data-w",      "116");
    NativeRenderer::set_attr(&ctrl, "data-h",      "44");

    let dec = thresh_btn("−", theme);
    let sa1 = Arc::clone(&state_arc);
    let c1  = champion.clone();
    NativeRenderer::on_event(&dec, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = sa1.lock() {
            let v = s.patch_lookback.saturating_sub(1).max(1);
            s.patch_lookback = v;
            let _ = s.fetch_tx.try_send(FetchCmd::ReloadNotes {
                depth: v, champion: c1.clone(),
            });
        }
    }));
    NativeRenderer::append(&ctrl, &dec);

    let val_div = NativeRenderer::element("div");
    NativeRenderer::set_attr(&val_div, "data-fill", theme.surface);
    NativeRenderer::set_attr(&val_div, "data-w",    "48");
    NativeRenderer::set_attr(&val_div, "data-h",    "44");
    let val_t = NativeRenderer::text(&format!(" {lookback}p "));
    NativeRenderer::set_attr(&val_t, "data-color",       theme.text);
    NativeRenderer::set_attr(&val_t, "data-text-size",   "15");
    NativeRenderer::set_attr(&val_t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&val_t, "data-h",           "44");
    NativeRenderer::set_attr(&val_t, "data-align",       "center");
    NativeRenderer::append(&val_div, &val_t);
    NativeRenderer::append(&ctrl, &val_div);

    let inc = thresh_btn("+", theme);
    let sa2 = Arc::clone(&state_arc);
    let c2  = champion.clone();
    NativeRenderer::on_event(&inc, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = sa2.lock() {
            let v = (s.patch_lookback + 1).min(10);
            s.patch_lookback = v;
            let _ = s.fetch_tx.try_send(FetchCmd::ReloadNotes {
                depth: v, champion: c2.clone(),
            });
        }
    }));
    NativeRenderer::append(&ctrl, &inc);

    NativeRenderer::append(parent, &ctrl);

    let pad = NativeRenderer::element("div");
    NativeRenderer::set_attr(&pad, "data-w", "12");
    NativeRenderer::append(parent, &pad);
}

fn patch_notes_tab(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    main_w:    u32,
    body_h:    u32,
) {
    const PATCH_SIDEBAR_W: u32 = 220;
    const SIDEBAR_HDR_H:   u32 = 52;
    const TAB_BAR_H:       u32 = 72;
    const OVERVIEW_H:      u32 = 220;
    let content_w = main_w.saturating_sub(PATCH_SIDEBAR_W);

    let outer = NativeRenderer::element("div");
    NativeRenderer::set_attr(&outer, "data-layout", "row");
    NativeRenderer::set_attr(&outer, "data-w",      &main_w.to_string());
    NativeRenderer::set_attr(&outer, "data-height", &body_h.to_string());
    NativeRenderer::append(parent, &outer);

    let selected_ver: Option<String> = state.selected_patch_ver.clone()
        .or_else(|| state.global_patches.first().map(|(v, _)| v.clone()));

    // Left: patch version sidebar
    {
        let sidebar = NativeRenderer::element("div");
        NativeRenderer::set_attr(&sidebar, "data-layout",       "column");
        NativeRenderer::set_attr(&sidebar, "data-w",            &PATCH_SIDEBAR_W.to_string());
        NativeRenderer::set_attr(&sidebar, "data-height",       &body_h.to_string());
        NativeRenderer::set_attr(&sidebar, "data-fill",         theme.surface);
        NativeRenderer::set_attr(&sidebar, "data-border-right", &format!("{}:1", theme.divider));
        NativeRenderer::append(&outer, &sidebar);

        let hdr = hrow(SIDEBAR_HDR_H);
        NativeRenderer::set_attr(&hdr, "data-fill",          theme.surface_hi);
        NativeRenderer::set_attr(&hdr, "data-border-bottom", &format!("{}:1", theme.divider));
        NativeRenderer::set_attr(&hdr, "data-pad",           "0 0 0 20");
        let hdr_t = NativeRenderer::text("PATCHES");
        NativeRenderer::set_attr(&hdr_t, "data-color",       theme.muted);
        NativeRenderer::set_attr(&hdr_t, "data-text-size",   "14");
        NativeRenderer::set_attr(&hdr_t, "data-text-weight", "bold");
        NativeRenderer::set_attr(&hdr_t, "data-height",           &SIDEBAR_HDR_H.to_string());
        NativeRenderer::append(&hdr, &hdr_t);
        NativeRenderer::append(&sidebar, &hdr);

        let list_h = body_h.saturating_sub(SIDEBAR_HDR_H);
        let list = NativeRenderer::element("div");
        NativeRenderer::set_attr(&list, "data-scroll-y",  "true");
        NativeRenderer::set_attr(&list, "data-scroll-id", "patch-sidebar-scroll");
        NativeRenderer::set_attr(&list, "data-height",    &list_h.to_string());
        NativeRenderer::set_attr(&list, "data-pad",       "0 0 0 8");
        NativeRenderer::append(&sidebar, &list);

        if state.global_patches.is_empty() {
            let msg = NativeRenderer::text(if state.global_loading { "Loading..." } else { "No data" });
            NativeRenderer::set_attr(&msg, "data-color",     theme.muted);
            NativeRenderer::set_attr(&msg, "data-text-size", "16");
            NativeRenderer::set_attr(&msg, "data-height",         "52");
            NativeRenderer::set_attr(&msg, "data-pad",       "0 0 0 20");
            NativeRenderer::append(&list, &msg);
        } else {
            for (patch_ver, patch_changes) in &state.global_patches {
                let is_sel = selected_ver.as_deref() == Some(patch_ver.as_str());

                // Container div is the hit target (text nodes return None from hit_test).
                let ver_btn = NativeRenderer::element("div");
                NativeRenderer::set_attr(&ver_btn, "data-height",      "64");
                NativeRenderer::set_attr(&ver_btn, "data-fill",        if is_sel { theme.surface_hi } else { "transparent" });
                NativeRenderer::set_attr(&ver_btn, "data-hover-fill",  theme.surface_hi);
                NativeRenderer::set_attr(&ver_btn, "data-pad",         "0 0 0 20");
                // Always show left border — thick accent for selected, thin divider for others.
                NativeRenderer::set_attr(&ver_btn, "data-border-left",
                    &if is_sel { format!("{}:4", theme.accent) } else { format!("{}:2", theme.divider) });
                let pv = patch_ver.clone();
                let sa = state_arc.clone();
                NativeRenderer::on_event(&ver_btn, "click", Box::new(move |_: BrickEvent| {
                    if let Ok(mut s) = sa.lock() { s.selected_patch_ver = Some(pv.clone()); }
                }));
                let ver_t = NativeRenderer::text(&format!("Patch {patch_ver}"));
                NativeRenderer::set_attr(&ver_t, "data-color",       if is_sel { theme.text } else { theme.muted });
                NativeRenderer::set_attr(&ver_t, "data-text-size",   "18");
                NativeRenderer::set_attr(&ver_t, "data-text-weight", "bold");
                NativeRenderer::set_attr(&ver_t, "data-height",      "64");
                NativeRenderer::append(&ver_btn, &ver_t);
                NativeRenderer::append(&list, &ver_btn);

                let ct_btn = NativeRenderer::element("div");
                NativeRenderer::set_attr(&ct_btn, "data-height",     "32");
                NativeRenderer::set_attr(&ct_btn, "data-fill",       if is_sel { theme.surface_hi } else { "transparent" });
                NativeRenderer::set_attr(&ct_btn, "data-hover-fill", theme.surface_hi);
                NativeRenderer::set_attr(&ct_btn, "data-pad",        "0 0 0 20");
                NativeRenderer::set_attr(&ct_btn, "data-border-left",
                    &if is_sel { format!("{}:4", theme.accent) } else { format!("{}:2", theme.divider) });
                let pv2 = patch_ver.clone();
                let sa2 = state_arc.clone();
                NativeRenderer::on_event(&ct_btn, "click", Box::new(move |_: BrickEvent| {
                    if let Ok(mut s) = sa2.lock() { s.selected_patch_ver = Some(pv2.clone()); }
                }));
                let ct = NativeRenderer::text(&format!("{} changes", patch_changes.len()));
                NativeRenderer::set_attr(&ct, "data-color",     theme.muted);
                NativeRenderer::set_attr(&ct, "data-text-size", "14");
                NativeRenderer::set_attr(&ct, "data-height",    "32");
                NativeRenderer::append(&ct_btn, &ct);
                NativeRenderer::append(&list, &ct_btn);

                // Gap between entries — carries the spine border for visual continuity.
                let gap_div = NativeRenderer::element("div");
                NativeRenderer::set_attr(&gap_div, "data-height",    "12");
                NativeRenderer::set_attr(&gap_div, "data-border-left",
                    &format!("{}:2", theme.divider));
                NativeRenderer::append(&list, &gap_div);
            }
        }
    }

    // Right: selected patch detail
    {
        let detail = NativeRenderer::element("div");
        NativeRenderer::set_attr(&detail, "data-layout", "column");
        NativeRenderer::set_attr(&detail, "data-w",      &content_w.to_string());
        NativeRenderer::set_attr(&detail, "data-height", &body_h.to_string());
        NativeRenderer::append(&outer, &detail);

        let patch_data = selected_ver.as_deref()
            .and_then(|ver| state.global_patches.iter().find(|(v, _)| v == ver));

        if state.global_loading {
            let msg = NativeRenderer::text("Fetching latest patch notes...");
            NativeRenderer::set_attr(&msg, "data-color",     theme.accent2);
            NativeRenderer::set_attr(&msg, "data-text-size", "18");
            NativeRenderer::set_attr(&msg, "data-height",         "52");
            NativeRenderer::set_attr(&msg, "data-pad",       "0 0 0 24");
            NativeRenderer::append(&detail, &msg);
        } else if let Some((ver, changes)) = patch_data {
            let champ_entries:  Vec<_> = changes.iter().filter(|c|  is_champion_entry(&c.patch)).collect();
            let rune_entries:   Vec<_> = changes.iter()
                .filter(|c| !is_champion_entry(&c.patch) &&  is_rune_entry(&c.patch)).collect();
            let rest:           Vec<_> = changes.iter()
                .filter(|c| !is_champion_entry(&c.patch) && !is_rune_entry(&c.patch)).collect();
            let item_entries:   Vec<_> = rest.iter().filter(|c| !is_patch_system_entry(&c.patch)).copied().collect();
            let system_entries: Vec<_> = rest.iter().filter(|c|  is_patch_system_entry(&c.patch)).copied().collect();

            let champ_buffs = champ_entries.iter().filter(|c| patch_change_class(&c.summary) == PatchClass::Buff).count();
            let champ_nerfs = champ_entries.iter().filter(|c| patch_change_class(&c.summary) == PatchClass::Nerf).count();
            let item_buffs  = item_entries.iter().filter(|c|  patch_change_class(&c.summary) == PatchClass::Buff).count();
            let item_nerfs  = item_entries.iter().filter(|c|  patch_change_class(&c.summary) == PatchClass::Nerf).count();

            let scroll_h = body_h.saturating_sub(OVERVIEW_H + TAB_BAR_H);

            // Overview card
            let card = NativeRenderer::element("div");
            NativeRenderer::set_attr(&card, "data-layout", "column");
            NativeRenderer::set_attr(&card, "data-w",      &content_w.to_string());
            NativeRenderer::set_attr(&card, "data-height", &OVERVIEW_H.to_string());
            NativeRenderer::set_attr(&card, "data-fill",   theme.surface);
            NativeRenderer::set_attr(&card, "data-border-bottom", &format!("{}:2", theme.accent));
            NativeRenderer::set_attr(&card, "data-pad",    "24 32 0 32");
            NativeRenderer::append(&detail, &card);

            let title = NativeRenderer::text(&format!("PATCH  {ver}"));
            NativeRenderer::set_attr(&title, "data-color",       theme.accent);
            NativeRenderer::set_attr(&title, "data-text-size",   "28");
            NativeRenderer::set_attr(&title, "data-text-weight", "bold");
            NativeRenderer::set_attr(&title, "data-height",           "44");
            NativeRenderer::append(&card, &title);

            spacer(16, &card);
            patch_overview_row("CHAMPIONS", champ_entries.len(), champ_buffs, champ_nerfs, theme, &card);
            spacer(10, &card);
            patch_overview_row("ITEMS",     item_entries.len(),  item_buffs,  item_nerfs,  theme, &card);
            spacer(10, &card);

            let foot = hrow(30);
            NativeRenderer::append(&card, &foot);
            let rune_t = NativeRenderer::text(&format!("{} rune changes", rune_entries.len()));
            NativeRenderer::set_attr(&rune_t, "data-color",     theme.muted);
            NativeRenderer::set_attr(&rune_t, "data-text-size", "15");
            NativeRenderer::set_attr(&rune_t, "data-height",         "30");
            NativeRenderer::append(&foot, &rune_t);
            if !system_entries.is_empty() {
                let sep = NativeRenderer::text("  |  ");
                NativeRenderer::set_attr(&sep, "data-color",     theme.muted);
                NativeRenderer::set_attr(&sep, "data-text-size", "15");
                NativeRenderer::set_attr(&sep, "data-height",         "30");
                NativeRenderer::append(&foot, &sep);
                let sys_t = NativeRenderer::text(&format!("{} system changes", system_entries.len()));
                NativeRenderer::set_attr(&sys_t, "data-color",     theme.muted);
                NativeRenderer::set_attr(&sys_t, "data-text-size", "15");
                NativeRenderer::set_attr(&sys_t, "data-height",         "30");
                NativeRenderer::append(&foot, &sys_t);
            }

            // Tab bar
            let tab_bar = hrow(TAB_BAR_H);
            NativeRenderer::set_attr(&tab_bar, "data-fill",          theme.surface);
            NativeRenderer::set_attr(&tab_bar, "data-border-bottom", &format!("{}:1", theme.divider));
            NativeRenderer::append(&detail, &tab_bar);

            for (dt, label, count) in [
                (PatchDetailTab::Champions, "CHAMPIONS", champ_entries.len()),
                (PatchDetailTab::Items,     "ITEMS",     item_entries.len()),
                (PatchDetailTab::Runes,     "RUNES",     rune_entries.len()),
                (PatchDetailTab::System,    "SYSTEM",    system_entries.len()),
            ] {
                let is_active = state.patch_detail_tab == dt;
                // Container div is the hit target (text nodes return None from hit_test).
                let lbl_btn = NativeRenderer::element("div");
                NativeRenderer::set_attr(&lbl_btn, "data-layout",      "row");
                NativeRenderer::set_attr(&lbl_btn, "data-height",      &TAB_BAR_H.to_string());
                NativeRenderer::set_attr(&lbl_btn, "data-w",           "190");
                NativeRenderer::set_attr(&lbl_btn, "data-fill",        if is_active { theme.surface_hi } else { "transparent" });
                NativeRenderer::set_attr(&lbl_btn, "data-hover-fill",  theme.surface_hi);
                if is_active {
                    NativeRenderer::set_attr(&lbl_btn, "data-border-bottom", &format!("{}:3", theme.accent));
                }
                let sa = state_arc.clone();
                NativeRenderer::on_event(&lbl_btn, "click", Box::new(move |_: BrickEvent| {
                    if let Ok(mut s) = sa.lock() { s.patch_detail_tab = dt; }
                }));
                let lbl = NativeRenderer::text(&format!("{label}  {count}"));
                NativeRenderer::set_attr(&lbl, "data-color",       if is_active { theme.text } else { theme.muted });
                NativeRenderer::set_attr(&lbl, "data-text-size",   "18");
                NativeRenderer::set_attr(&lbl, "data-text-weight", "bold");
                NativeRenderer::set_attr(&lbl, "data-height",      &TAB_BAR_H.to_string());
                NativeRenderer::set_attr(&lbl, "data-align",       "center");
                NativeRenderer::append(&lbl_btn, &lbl);
                NativeRenderer::append(&tab_bar, &lbl_btn);
            }

            // Scrollable entry list
            let scroll = NativeRenderer::element("div");
            NativeRenderer::set_attr(&scroll, "data-scroll-y",  "true");
            NativeRenderer::set_attr(&scroll, "data-scroll-id", "patch-detail-scroll");
            NativeRenderer::set_attr(&scroll, "data-layout",    "column");
            NativeRenderer::set_attr(&scroll, "data-height",    &scroll_h.to_string());
            NativeRenderer::append(&detail, &scroll);

            let active: Vec<&PatchChange> = match state.patch_detail_tab {
                PatchDetailTab::Champions => champ_entries.clone(),
                PatchDetailTab::Items     => item_entries.clone(),
                PatchDetailTab::Runes     => rune_entries.clone(),
                PatchDetailTab::System    => system_entries.clone(),
            };

            if active.is_empty() {
                let none_t = NativeRenderer::text("No changes in this category.");
                NativeRenderer::set_attr(&none_t, "data-color",     theme.muted);
                NativeRenderer::set_attr(&none_t, "data-text-size", "16");
                NativeRenderer::set_attr(&none_t, "data-height",         "52");
                NativeRenderer::set_attr(&none_t, "data-pad",       "0 0 0 24");
                NativeRenderer::append(&scroll, &none_t);
            } else {
                let icon_style = match state.patch_detail_tab {
                    PatchDetailTab::Champions => EntryIconStyle::Champion,
                    PatchDetailTab::Items     => EntryIconStyle::Item,
                    PatchDetailTab::Runes     => EntryIconStyle::Rune,
                    PatchDetailTab::System    => EntryIconStyle::None,
                };
                for change in active {
                    patch_entry_row(change, icon_style, theme, &scroll);
                }
            }
        } else {
            let msg = NativeRenderer::text("No patch data available. Patch notes are fetched on startup.");
            NativeRenderer::set_attr(&msg, "data-color",     theme.muted);
            NativeRenderer::set_attr(&msg, "data-text-size", "16");
            NativeRenderer::set_attr(&msg, "data-height",         "52");
            NativeRenderer::set_attr(&msg, "data-pad",       "0 24 0 24");
            NativeRenderer::append(&detail, &msg);
        }
    }
}

/// Sticky category divider inside a scrollable patch notes list.
fn patch_section_divider(label: &str, count: usize, theme: &Theme, parent: &NativeNode) {
    let row = hrow(42);
    NativeRenderer::set_attr(&row, "data-pad",         "0 0 0 16");
    NativeRenderer::set_attr(&row, "data-fill",        theme.surface);
    NativeRenderer::set_attr(&row, "data-border-left", &format!("{}:3", theme.accent));
    NativeRenderer::set_attr(&row, "data-border-bottom", &format!("{}:1", theme.divider));

    let t = NativeRenderer::text(label);
    NativeRenderer::set_attr(&t, "data-color",       theme.accent);
    NativeRenderer::set_attr(&t, "data-text-size",   "16");
    NativeRenderer::set_attr(&t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&t, "data-height",           "42");
    NativeRenderer::set_attr(&t, "data-flex",        "1.0");
    NativeRenderer::append(&row, &t);

    if count > 0 {
        let cnt = NativeRenderer::text(&format!("{count}  "));
        NativeRenderer::set_attr(&cnt, "data-color",     theme.muted);
        NativeRenderer::set_attr(&cnt, "data-text-size", "14");
        NativeRenderer::set_attr(&cnt, "data-h",         "42");
        NativeRenderer::append(&row, &cnt);
    }

    NativeRenderer::append(parent, &row);
}

fn patch_change_block(
    header_text: &str,
    summary:     &str,
    key:         &str,
    expanded:    bool,
    state_arc:   Arc<Mutex<ControlPanelState>>,
    theme:       &Theme,
    parent:      &NativeNode,
) {
    let change_count = summary.lines().filter(|l| !l.trim().is_empty()).count();

    // Clickable header row — chevron indicates expand/collapse state.
    let header = hrow(36);
    NativeRenderer::set_attr(&header, "data-fill",         "#0c0f1a");
    NativeRenderer::set_attr(&header, "data-border-left",  &format!("{}:2", theme.accent2));
    NativeRenderer::set_attr(&header, "data-border-bottom",&format!("{}:1", theme.divider));
    NativeRenderer::set_attr(&header, "data-hover-fill",   "#0f1220");
    NativeRenderer::set_attr(&header, "data-pad",          "0 0 0 16");

    let chev = NativeRenderer::text(if expanded { "▼  " } else { "▶  " });
    NativeRenderer::set_attr(&chev, "data-color",     theme.accent2);
    NativeRenderer::set_attr(&chev, "data-text-size", "13");
    NativeRenderer::set_attr(&chev, "data-h",         "36");
    NativeRenderer::append(&header, &chev);

    let ver = NativeRenderer::text(header_text);
    NativeRenderer::set_attr(&ver, "data-color",       theme.accent2);
    NativeRenderer::set_attr(&ver, "data-text-size",   "14");
    NativeRenderer::set_attr(&ver, "data-text-weight", "bold");
    NativeRenderer::set_attr(&ver, "data-h",           "36");
    NativeRenderer::set_attr(&ver, "data-flex",        "1.0");
    NativeRenderer::append(&header, &ver);

    if !expanded {
        let change_type = infer_change_type(summary);
        let (type_text, type_color) = match change_type {
            PatchType::Buff   => ("▲ BUFF  ",   theme.ok),
            PatchType::Nerf   => ("▼ NERF  ",   theme.err),
            PatchType::Adjust => ("◆ ADJUST  ", theme.accent2),
        };
        let type_lbl = NativeRenderer::text(type_text);
        NativeRenderer::set_attr(&type_lbl, "data-color",       type_color);
        NativeRenderer::set_attr(&type_lbl, "data-text-size",   "13");
        NativeRenderer::set_attr(&type_lbl, "data-text-weight", "bold");
        NativeRenderer::set_attr(&type_lbl, "data-h",           "36");
        NativeRenderer::append(&header, &type_lbl);

        let count_lbl = NativeRenderer::text(&format!("{change_count}  "));
        NativeRenderer::set_attr(&count_lbl, "data-color",     theme.muted);
        NativeRenderer::set_attr(&count_lbl, "data-text-size", "13");
        NativeRenderer::set_attr(&count_lbl, "data-h",         "36");
        NativeRenderer::append(&header, &count_lbl);
    }

    let key_owned = key.to_string();
    NativeRenderer::on_event(&header, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = state_arc.lock() {
            if s.patch_expanded.contains(&key_owned) {
                s.patch_expanded.remove(&key_owned);
            } else {
                s.patch_expanded.insert(key_owned.clone());
            }
        }
    }));
    NativeRenderer::append(parent, &header);

    if !expanded { return; }

    let mut seen_ability_header = false;
    for line in summary.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        match classify_line(trimmed) {
            LineKind::AbilityHeader => {
                if seen_ability_header { spacer(4, parent); }
                seen_ability_header = true;
                patch_ability_header_row(trimmed, theme, parent);
            }
            LineKind::NewFeature => patch_new_feature_row(trimmed, theme, parent),
            LineKind::Removed    => patch_removed_row(trimmed, theme, parent),
            LineKind::StatChange => patch_stat_change_row(trimmed, theme, parent),
            LineKind::Text if !seen_ability_header => patch_flavor_row(trimmed, theme, parent),
            LineKind::Text => patch_plain_row(trimmed, theme, parent),
        }
    }
    spacer(8, parent);
}

/// Extract the first numeric value (integer or decimal) from a string.
fn extract_first_number(s: &str) -> Option<f64> {
    let start = s.find(|c: char| c.is_ascii_digit())?;
    let rest = &s[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Choose a colour for the "after" value in a stat change row.
/// Green = buff, red = nerf. Accounts for stats where higher values are bad
/// (cooldowns, costs, cast times, etc.).
fn stat_direction_color<'t>(before: &str, after: &str, theme: &'t Theme) -> &'t str {
    let label_lower = before.to_lowercase();
    let nerf_when_higher = label_lower.contains("cooldown")
        || label_lower.contains(" cost")
        || label_lower.contains("cast time")
        || label_lower.contains("wind-up")
        || label_lower.contains("windup")
        || label_lower.contains("delay");

    let search_in = before.splitn(2, ':').nth(1).unwrap_or(before);
    let old_val = extract_first_number(search_in);
    let new_val = extract_first_number(after);

    match (old_val, new_val) {
        (Some(old), Some(new)) if (old - new).abs() > f64::EPSILON => {
            let is_buff = (new > old) ^ nerf_when_higher;
            if is_buff { theme.ok } else { theme.err }
        }
        _ => theme.text,
    }
}

/// "REMOVED Old Mechanic" — red badge + description.
fn patch_removed_row(line: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(26);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 32");
    NativeRenderer::set_attr(&row, "data-fill", "#1a0a0a");

    let badge = NativeRenderer::text("✕ REMOVED  ");
    NativeRenderer::set_attr(&badge, "data-color", theme.err);
    NativeRenderer::set_attr(&badge, "data-text-size", "15");
    NativeRenderer::set_attr(&badge, "data-text-weight", "bold");
    NativeRenderer::set_attr(&badge, "data-h", "26");
    NativeRenderer::set_attr(&badge, "data-flex", "0.14");
    NativeRenderer::append(&row, &badge);

    let desc = line.strip_prefix("REMOVED ").unwrap_or(line);
    let t = NativeRenderer::text(desc);
    NativeRenderer::set_attr(&t, "data-color", theme.err);
    NativeRenderer::set_attr(&t, "data-text-size", "16");
    NativeRenderer::set_attr(&t, "data-h", "26");
    NativeRenderer::set_attr(&t, "data-flex", "0.86");
    NativeRenderer::append(&row, &t);

    NativeRenderer::append(parent, &row);
}

/// "W - Safeguard" / "Base Stats" section divider.
fn patch_ability_header_row(text: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(30);
    NativeRenderer::set_attr(&row, "data-fill", theme.surface);
    NativeRenderer::set_attr(&row, "data-border-left", &format!("{}:2", theme.accent2));
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    let t = NativeRenderer::text(text);
    NativeRenderer::set_attr(&t, "data-color", theme.accent2);
    NativeRenderer::set_attr(&t, "data-text-size", "16");
    NativeRenderer::set_attr(&t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&t, "data-h", "30");
    NativeRenderer::append(&row, &t);
    NativeRenderer::append(parent, &row);
}

// ── Change-type inference ─────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum PatchType { Buff, Nerf, Adjust }

fn stat_change_direction(before: &str, after: &str) -> Option<bool> {
    let label_lower = before.to_lowercase();
    let nerf_when_higher = label_lower.contains("cooldown")
        || label_lower.contains(" cost")
        || label_lower.contains("cast time")
        || label_lower.contains("wind-up")
        || label_lower.contains("windup")
        || label_lower.contains("delay");
    let search_in = before.splitn(2, ':').nth(1).unwrap_or(before);
    let old_val = extract_first_number(search_in);
    let new_val = extract_first_number(after);
    match (old_val, new_val) {
        (Some(old), Some(new)) if (old - new).abs() > f64::EPSILON => {
            Some((new > old) ^ nerf_when_higher)
        }
        _ => None,
    }
}

fn infer_change_type(summary: &str) -> PatchType {
    let mut buffs = 0i32;
    let mut nerfs = 0i32;
    for line in summary.lines() {
        let l = line.trim();
        if l.starts_with("NEW ") { buffs += 2; continue; }
        if l.starts_with("REMOVED ") { nerfs += 2; continue; }
        if let Some(arrow_pos) = l.find('⇒') {
            let before = l[..arrow_pos].trim_end();
            let after  = l[arrow_pos + '⇒'.len_utf8()..].trim_start();
            if let Some(is_buff) = stat_change_direction(before, after) {
                if is_buff { buffs += 1; } else { nerfs += 1; }
            }
        }
    }
    if buffs == 0 && nerfs == 0 { return PatchType::Adjust; }
    let ratio = buffs as f64 / (buffs + nerfs).max(1) as f64;
    if ratio >= 0.65 { PatchType::Buff } else if ratio <= 0.35 { PatchType::Nerf } else { PatchType::Adjust }
}

/// "Shield: 70/115... ⇒ 60/105..." — muted before, accent arrow, bright after.
fn patch_stat_change_row(line: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(26);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 32");

    if let Some(arrow_pos) = line.find('⇒') {
        let before = line[..arrow_pos].trim_end();
        let after  = line[arrow_pos + '⇒'.len_utf8()..].trim_start();
        let after_color = stat_direction_color(before, after, theme);

        let b = NativeRenderer::text(before);
        NativeRenderer::set_attr(&b, "data-color", theme.muted);
        NativeRenderer::set_attr(&b, "data-text-size", "16");
        NativeRenderer::set_attr(&b, "data-h", "26");
        NativeRenderer::set_attr(&b, "data-flex", "0.53");
        NativeRenderer::append(&row, &b);

        let arr = NativeRenderer::text(" ⇒ ");
        NativeRenderer::set_attr(&arr, "data-color", theme.accent);
        NativeRenderer::set_attr(&arr, "data-text-size", "16");
        NativeRenderer::set_attr(&arr, "data-h", "26");
        NativeRenderer::set_attr(&arr, "data-flex", "0.05");
        NativeRenderer::append(&row, &arr);

        let a = NativeRenderer::text(after);
        NativeRenderer::set_attr(&a, "data-color", after_color);
        NativeRenderer::set_attr(&a, "data-text-size", "16");
        NativeRenderer::set_attr(&a, "data-h", "26");
        NativeRenderer::set_attr(&a, "data-flex", "0.42");
        NativeRenderer::append(&row, &a);
    } else {
        let t = NativeRenderer::text(line);
        NativeRenderer::set_attr(&t, "data-color", theme.text);
        NativeRenderer::set_attr(&t, "data-text-size", "16");
        NativeRenderer::set_attr(&t, "data-h", "26");
        NativeRenderer::append(&row, &t);
    }

    NativeRenderer::append(parent, &row);
}

/// "NEW Shield on non-champions: …" — green badge + description.
fn patch_new_feature_row(line: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(26);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 32");
    NativeRenderer::set_attr(&row, "data-fill", "#0a1a0d");

    let badge = NativeRenderer::text("◆ NEW  ");
    NativeRenderer::set_attr(&badge, "data-color", theme.ok);
    NativeRenderer::set_attr(&badge, "data-text-size", "15");
    NativeRenderer::set_attr(&badge, "data-text-weight", "bold");
    NativeRenderer::set_attr(&badge, "data-h", "26");
    NativeRenderer::set_attr(&badge, "data-flex", "0.12");
    NativeRenderer::append(&row, &badge);

    let desc = line.strip_prefix("NEW ").unwrap_or(line);
    let t = NativeRenderer::text(desc);
    NativeRenderer::set_attr(&t, "data-color", theme.ok);
    NativeRenderer::set_attr(&t, "data-text-size", "16");
    NativeRenderer::set_attr(&t, "data-h", "26");
    NativeRenderer::set_attr(&t, "data-flex", "0.88");
    NativeRenderer::append(&row, &t);

    NativeRenderer::append(parent, &row);
}

/// Context quote before the first ability section — muted, smaller.
fn patch_flavor_row(text: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(24);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    let t = NativeRenderer::text(text);
    NativeRenderer::set_attr(&t, "data-color", theme.muted);
    NativeRenderer::set_attr(&t, "data-text-size", "15");
    NativeRenderer::set_attr(&t, "data-h", "24");
    NativeRenderer::append(&row, &t);
    NativeRenderer::append(parent, &row);
}

/// Plain bullet inside an ability section.
fn patch_plain_row(text: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(26);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 32");
    let t = NativeRenderer::text(text);
    NativeRenderer::set_attr(&t, "data-color", theme.text);
    NativeRenderer::set_attr(&t, "data-text-size", "16");
    NativeRenderer::set_attr(&t, "data-h", "26");
    NativeRenderer::append(&row, &t);
    NativeRenderer::append(parent, &row);
}

// ── Automation tab ────────────────────────────────────────────────────────────

fn automation_tab(
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    body_h:    u32,
) {
    // ── Picker mode: show a full-panel champion grid for the selected slot ─────
    if let Some((ref role, is_ban)) = state.configuring_slot {
        render_roster_picker(role, is_ban, state, state_arc, theme, parent, body_h);
        return;
    }

    // ── Normal view: automation toggles + champion roster ──────────────────────
    let cfg = state.automation.lock()
        .map(|g| g.clone())
        .unwrap_or_default();

    let scroll = NativeRenderer::element("div");
    NativeRenderer::set_attr(&scroll, "data-scroll-y", "true");
    NativeRenderer::set_attr(&scroll, "data-height", &body_h.to_string());

    section_header(
        "PRE-GAME AUTOMATION",
        "Toggles apply on the next 500 ms poll — no restart needed.",
        theme, &scroll,
    );

    auto_group_header("QUEUE", theme, &scroll);
    auto_toggle(
        "Accept ready-check",
        "Clicks Accept the moment a match-found popup appears.",
        cfg.auto_accept_queue,
        Arc::clone(&state.automation),
        Arc::new(|c: &mut AutomationConfig| c.auto_accept_queue = !c.auto_accept_queue),
        theme, &scroll,
    );

    spacer(8, &scroll);
    auto_group_header("BAN PHASE", theme, &scroll);
    auto_toggle(
        "Hover recommended ban",
        "Instantly selects the top meta ban when the ban timer starts.",
        cfg.auto_hover_ban,
        Arc::clone(&state.automation),
        Arc::new(|c: &mut AutomationConfig| c.auto_hover_ban = !c.auto_hover_ban),
        theme, &scroll,
    );
    auto_toggle(
        "Confirm ban at deadline",
        &format!("Submits the hovered ban when timer drops below {:.0}s.", cfg.lock_threshold_secs),
        cfg.auto_confirm_ban,
        Arc::clone(&state.automation),
        Arc::new(|c: &mut AutomationConfig| c.auto_confirm_ban = !c.auto_confirm_ban),
        theme, &scroll,
    );

    spacer(8, &scroll);
    auto_group_header("PICK PHASE", theme, &scroll);
    auto_toggle(
        "Hover recommended pick",
        "Selects the top role pick if you haven't hovered anything yet.",
        cfg.auto_hover_pick,
        Arc::clone(&state.automation),
        Arc::new(|c: &mut AutomationConfig| c.auto_hover_pick = !c.auto_hover_pick),
        theme, &scroll,
    );
    auto_toggle(
        "Lock in at deadline",
        &format!("Locks the hovered champion when timer drops below {:.0}s.", cfg.lock_threshold_secs),
        cfg.auto_lock_in,
        Arc::clone(&state.automation),
        Arc::new(|c: &mut AutomationConfig| c.auto_lock_in = !c.auto_lock_in),
        theme, &scroll,
    );

    spacer(8, &scroll);
    auto_group_header("RUNES & ITEMS", theme, &scroll);
    auto_toggle(
        "Auto-import runes",
        "Applies recommended rune page + summoner spells once per session when you hover a champion.",
        cfg.auto_import_runes,
        Arc::clone(&state.automation),
        Arc::new(|c: &mut AutomationConfig| c.auto_import_runes = !c.auto_import_runes),
        theme, &scroll,
    );

    spacer(16, &scroll);

    // Lock-threshold adjuster.
    let thresh_row = hrow(44);
    NativeRenderer::set_attr(&thresh_row, "data-pad", "0 0 0 20");

    let label = NativeRenderer::text("Lock / confirm threshold");
    NativeRenderer::set_attr(&label, "data-color", theme.muted);
    NativeRenderer::set_attr(&label, "data-text-size", "16");
    NativeRenderer::set_attr(&label, "data-flex", "1.0");
    NativeRenderer::set_attr(&label, "data-h", "44");
    NativeRenderer::append(&thresh_row, &label);

    let ctrl = NativeRenderer::element("div");
    NativeRenderer::set_attr(&ctrl, "data-layout", "row");
    NativeRenderer::set_attr(&ctrl, "data-w", "128");
    NativeRenderer::set_attr(&ctrl, "data-h", "44");

    let dec_btn = thresh_btn("−", theme);
    let dec_arc = Arc::clone(&state.automation);
    NativeRenderer::on_event(&dec_btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut c) = dec_arc.lock() {
            c.lock_threshold_secs = (c.lock_threshold_secs - 1.0).max(1.0);
        }
    }));
    NativeRenderer::append(&ctrl, &dec_btn);

    let val_div = NativeRenderer::element("div");
    NativeRenderer::set_attr(&val_div, "data-fill", theme.surface);
    NativeRenderer::set_attr(&val_div, "data-w", "56");
    NativeRenderer::set_attr(&val_div, "data-h", "44");
    let val_txt = NativeRenderer::text(&format!(" {:.0}s ", cfg.lock_threshold_secs));
    NativeRenderer::set_attr(&val_txt, "data-color", theme.text);
    NativeRenderer::set_attr(&val_txt, "data-text-size", "17");
    NativeRenderer::set_attr(&val_txt, "data-text-weight", "bold");
    NativeRenderer::set_attr(&val_txt, "data-h", "44");
    NativeRenderer::set_attr(&val_txt, "data-align", "center");
    NativeRenderer::append(&val_div, &val_txt);
    NativeRenderer::append(&ctrl, &val_div);

    let inc_btn = thresh_btn("+", theme);
    let inc_arc = Arc::clone(&state.automation);
    NativeRenderer::on_event(&inc_btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut c) = inc_arc.lock() {
            c.lock_threshold_secs = (c.lock_threshold_secs + 1.0).min(15.0);
        }
    }));
    NativeRenderer::append(&ctrl, &inc_btn);

    NativeRenderer::append(&thresh_row, &ctrl);
    let rpad = NativeRenderer::element("div");
    NativeRenderer::set_attr(&rpad, "data-w", "20");
    NativeRenderer::append(&thresh_row, &rpad);
    NativeRenderer::append(&scroll, &thresh_row);

    // ── Champion roster ────────────────────────────────────────────────────────
    spacer(16, &scroll);
    auto_group_header("CHAMPION ROSTER", theme, &scroll);

    for &(role_key, role_label) in &[
        ("top",     "TOP"),
        ("jungle",  "JUNGLE"),
        ("middle",  "MID"),
        ("bottom",  "BOT"),
        ("utility", "SUPPORT"),
    ] {
        let pick_champ = cfg.pick_champions.get(role_key).cloned();
        let ban_champ  = cfg.ban_champions.get(role_key).cloned();
        role_champ_row(
            role_key, role_label,
            pick_champ.as_deref(), ban_champ.as_deref(),
            Arc::clone(&state_arc),
            theme, &scroll,
        );
    }

    spacer(24, &scroll);
    NativeRenderer::append(parent, &scroll);
}

/// Full-panel champion picker shown when the user clicks a roster slot.
fn render_roster_picker(
    role:      &str,
    is_ban:    bool,
    state:     &ControlPanelState,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
    parent:    &NativeNode,
    body_h:    u32,
) {
    let kind_label  = if is_ban { "BAN" } else { "PICK" };
    let role_label  = match role {
        "top"     => "TOP",
        "jungle"  => "JUNGLE",
        "middle"  => "MID",
        "bottom"  => "BOT",
        "utility" => "SUPPORT",
        other     => other,
    };

    // Header: [← BACK]  PICK for TOP  [CLEAR]
    let header = hrow(60);
    NativeRenderer::set_attr(&header, "data-fill", theme.surface);
    NativeRenderer::set_attr(&header, "data-border-bottom", &format!("{}:1", theme.divider));
    NativeRenderer::set_attr(&header, "data-pad", "0 0 0 16");

    let back_btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&back_btn, "data-fill",       theme.surface_hi);
    NativeRenderer::set_attr(&back_btn, "data-w",          "88");
    NativeRenderer::set_attr(&back_btn, "data-h",          "60");
    NativeRenderer::set_attr(&back_btn, "data-hover-fill", "#1a1a28");
    let back_txt = NativeRenderer::text("← BACK");
    NativeRenderer::set_attr(&back_txt, "data-color",       theme.accent);
    NativeRenderer::set_attr(&back_txt, "data-text-size",   "14");
    NativeRenderer::set_attr(&back_txt, "data-text-weight", "bold");
    NativeRenderer::set_attr(&back_txt, "data-h",           "60");
    NativeRenderer::set_attr(&back_txt, "data-align",       "center");
    NativeRenderer::append(&back_btn, &back_txt);
    let sa_back = Arc::clone(&state_arc);
    NativeRenderer::on_event(&back_btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut s) = sa_back.lock() { s.configuring_slot = None; }
    }));
    NativeRenderer::append(&header, &back_btn);

    let title_t = NativeRenderer::text(&format!("  {kind_label} for {role_label}"));
    NativeRenderer::set_attr(&title_t, "data-color",       theme.text);
    NativeRenderer::set_attr(&title_t, "data-text-size",   "18");
    NativeRenderer::set_attr(&title_t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&title_t, "data-h",           "60");
    NativeRenderer::set_attr(&title_t, "data-flex",        "1.0");
    NativeRenderer::append(&header, &title_t);

    let clear_btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&clear_btn, "data-fill",       "#1a0a0a");
    NativeRenderer::set_attr(&clear_btn, "data-w",          "88");
    NativeRenderer::set_attr(&clear_btn, "data-h",          "60");
    NativeRenderer::set_attr(&clear_btn, "data-hover-fill", "#2a0a0a");
    let clear_txt = NativeRenderer::text("CLEAR");
    NativeRenderer::set_attr(&clear_txt, "data-color",     theme.err);
    NativeRenderer::set_attr(&clear_txt, "data-text-size", "14");
    NativeRenderer::set_attr(&clear_txt, "data-h",         "60");
    NativeRenderer::set_attr(&clear_txt, "data-align",     "center");
    NativeRenderer::append(&clear_btn, &clear_txt);
    let sa_clear    = Arc::clone(&state_arc);
    let auto_clear  = Arc::clone(&state.automation);
    let role_clear  = role.to_string();
    NativeRenderer::on_event(&clear_btn, "click", Box::new(move |_: BrickEvent| {
        if let Ok(mut c) = auto_clear.lock() {
            if is_ban { c.ban_champions.remove(&role_clear); }
            else       { c.pick_champions.remove(&role_clear); }
        }
        if let Ok(mut s) = sa_clear.lock() { s.configuring_slot = None; }
    }));
    NativeRenderer::append(&header, &clear_btn);
    NativeRenderer::append(parent, &header);

    // Champion grid — 15 columns × 80 px, scrollable.
    const COLS: usize = 15;
    const CELL: u32   = 80;
    let grid_h = body_h.saturating_sub(60);
    let scroll = NativeRenderer::element("div");
    NativeRenderer::set_attr(&scroll, "data-scroll-y", "true");
    NativeRenderer::set_attr(&scroll, "data-height",   &grid_h.to_string());

    let auto_arc = Arc::clone(&state.automation);
    let role_str = role.to_string();

    for chunk in ALL_CHAMPIONS.chunks(COLS) {
        let row = hrow(CELL);
        for &name in chunk {
            let cell = NativeRenderer::element("div");
            NativeRenderer::set_attr(&cell, "data-w",          &CELL.to_string());
            NativeRenderer::set_attr(&cell, "data-height",     &CELL.to_string());
            NativeRenderer::set_attr(&cell, "data-image",      &format!("assets/champion_icons/{name}.png"));
            NativeRenderer::set_attr(&cell, "data-hover-fill", "#1a1a2e");

            let sa  = Arc::clone(&state_arc);
            let aa  = Arc::clone(&auto_arc);
            let r   = role_str.clone();
            let n   = name.to_string();
            NativeRenderer::on_event(&cell, "click", Box::new(move |_: BrickEvent| {
                if let Ok(mut c) = aa.lock() {
                    if is_ban { c.ban_champions.insert(r.clone(), n.clone()); }
                    else       { c.pick_champions.insert(r.clone(), n.clone()); }
                }
                if let Ok(mut s) = sa.lock() { s.configuring_slot = None; }
            }));
            NativeRenderer::append(&row, &cell);
        }
        NativeRenderer::append(&scroll, &row);
    }
    NativeRenderer::append(parent, &scroll);
}

/// One row in the CHAMPION ROSTER section: role label + pick slot + ban slot.
fn role_champ_row(
    role_key:   &str,
    role_label: &str,
    pick_champ: Option<&str>,
    ban_champ:  Option<&str>,
    state_arc:  Arc<Mutex<ControlPanelState>>,
    theme:      &Theme,
    parent:     &NativeNode,
) {
    let row = hrow(56);
    NativeRenderer::set_attr(&row, "data-fill", theme.surface);
    NativeRenderer::set_attr(&row, "data-pad",  "0 0 0 20");

    let lbl = NativeRenderer::text(role_label);
    NativeRenderer::set_attr(&lbl, "data-color",       theme.accent2);
    NativeRenderer::set_attr(&lbl, "data-text-size",   "15");
    NativeRenderer::set_attr(&lbl, "data-text-weight", "bold");
    NativeRenderer::set_attr(&lbl, "data-h",           "56");
    NativeRenderer::set_attr(&lbl, "data-w",           "88");
    NativeRenderer::append(&row, &lbl);

    let pick_btn = champ_slot_btn("PICK", pick_champ, role_key, false,
        Arc::clone(&state_arc), theme);
    NativeRenderer::append(&row, &pick_btn);

    let gap = NativeRenderer::element("div");
    NativeRenderer::set_attr(&gap, "data-w", "12");
    NativeRenderer::append(&row, &gap);

    let ban_btn = champ_slot_btn("BAN", ban_champ, role_key, true,
        Arc::clone(&state_arc), theme);
    NativeRenderer::append(&row, &ban_btn);

    NativeRenderer::append(parent, &row);

    let sep = NativeRenderer::element("div");
    NativeRenderer::set_attr(&sep, "data-h",   "1");
    NativeRenderer::set_attr(&sep, "data-fill", theme.divider);
    NativeRenderer::append(parent, &sep);
}

/// Slot button showing a configured champion (icon + name) or an empty placeholder.
/// Clicking always opens the picker for that slot.
fn champ_slot_btn(
    kind:      &str,
    champ:     Option<&str>,
    role_key:  &str,
    is_ban:    bool,
    state_arc: Arc<Mutex<ControlPanelState>>,
    theme:     &Theme,
) -> NativeNode {
    let slot_color = if is_ban { theme.err } else { theme.ok };

    let btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&btn, "data-layout",      "row");
    NativeRenderer::set_attr(&btn, "data-w",           "200");
    NativeRenderer::set_attr(&btn, "data-h",           "56");
    NativeRenderer::set_attr(&btn, "data-fill",        theme.surface_hi);
    NativeRenderer::set_attr(&btn, "data-hover-fill",  "#1a1a28");
    NativeRenderer::set_attr(&btn, "data-border-left", &format!("{}:2", slot_color));

    if let Some(name) = champ {
        let icon = NativeRenderer::element("div");
        NativeRenderer::set_attr(&icon, "data-w",      "44");
        NativeRenderer::set_attr(&icon, "data-height", "44");
        NativeRenderer::set_attr(&icon, "data-image",  &format!("assets/champion_icons/{name}.png"));

        let col = NativeRenderer::element("div");
        NativeRenderer::set_attr(&col, "data-layout", "column");
        NativeRenderer::set_attr(&col, "data-flex",   "1.0");
        NativeRenderer::set_attr(&col, "data-pad",    "0 0 0 10");

        let kind_t = NativeRenderer::text(kind);
        NativeRenderer::set_attr(&kind_t, "data-color",     slot_color);
        NativeRenderer::set_attr(&kind_t, "data-text-size", "12");
        NativeRenderer::set_attr(&kind_t, "data-h",         "22");
        NativeRenderer::append(&col, &kind_t);

        let name_t = NativeRenderer::text(name);
        NativeRenderer::set_attr(&name_t, "data-color",       theme.text);
        NativeRenderer::set_attr(&name_t, "data-text-size",   "15");
        NativeRenderer::set_attr(&name_t, "data-text-weight", "bold");
        NativeRenderer::set_attr(&name_t, "data-h",           "22");
        NativeRenderer::append(&col, &name_t);

        NativeRenderer::append(&btn, &icon);
        NativeRenderer::append(&btn, &col);

        // Attach handler to all non-text divs — brick has no event bubbling.
        let r1 = role_key.to_string();
        let sa1 = Arc::clone(&state_arc);
        NativeRenderer::on_event(&icon, "click", Box::new(move |_: BrickEvent| {
            if let Ok(mut s) = sa1.lock() { s.configuring_slot = Some((r1.clone(), is_ban)); }
        }));
        let r2 = role_key.to_string();
        let sa2 = Arc::clone(&state_arc);
        NativeRenderer::on_event(&col, "click", Box::new(move |_: BrickEvent| {
            if let Ok(mut s) = sa2.lock() { s.configuring_slot = Some((r2.clone(), is_ban)); }
        }));
        let r3 = role_key.to_string();
        let sa3 = Arc::clone(&state_arc);
        NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
            if let Ok(mut s) = sa3.lock() { s.configuring_slot = Some((r3.clone(), is_ban)); }
        }));
    } else {
        let ph = NativeRenderer::element("div");
        NativeRenderer::set_attr(&ph, "data-flex", "1.0");
        NativeRenderer::set_attr(&ph, "data-pad",  "0 0 0 12");

        let t = NativeRenderer::text(&format!("{kind}  ·  set champion"));
        NativeRenderer::set_attr(&t, "data-color",     theme.muted);
        NativeRenderer::set_attr(&t, "data-text-size", "15");
        NativeRenderer::set_attr(&t, "data-h",         "56");
        NativeRenderer::append(&ph, &t);
        NativeRenderer::append(&btn, &ph);

        let r1 = role_key.to_string();
        let sa1 = Arc::clone(&state_arc);
        NativeRenderer::on_event(&ph, "click", Box::new(move |_: BrickEvent| {
            if let Ok(mut s) = sa1.lock() { s.configuring_slot = Some((r1.clone(), is_ban)); }
        }));
        let r2 = role_key.to_string();
        let sa2 = Arc::clone(&state_arc);
        NativeRenderer::on_event(&btn, "click", Box::new(move |_: BrickEvent| {
            if let Ok(mut s) = sa2.lock() { s.configuring_slot = Some((r2.clone(), is_ban)); }
        }));
    }

    btn
}

fn auto_group_header(label: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(36);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    NativeRenderer::set_attr(&row, "data-fill", "#0c0f1a");
    NativeRenderer::set_attr(&row, "data-border-left", &format!("{}:3", theme.accent2));

    let t = NativeRenderer::text(label);
    NativeRenderer::set_attr(&t, "data-color", theme.accent2);
    NativeRenderer::set_attr(&t, "data-text-size", "16");
    NativeRenderer::set_attr(&t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&t, "data-h", "36");
    NativeRenderer::append(&row, &t);
    NativeRenderer::append(parent, &row);
}

fn auto_toggle(
    name:     &str,
    detail:   &str,
    enabled:  bool,
    auto_arc: Arc<Mutex<AutomationConfig>>,
    toggle:   Arc<dyn Fn(&mut AutomationConfig) + Send + Sync>,
    theme:    &Theme,
    parent:   &NativeNode,
) {
    let row = hrow(54);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    NativeRenderer::set_attr(
        &row, "data-fill",
        if enabled { theme.surface_hi } else { theme.surface },
    );
    NativeRenderer::set_attr(
        &row, "data-border-left",
        &format!("{}:4", if enabled { "#44ff66" } else { theme.muted }),
    );

    // Name + description — fills all space before the pill.
    let info = NativeRenderer::element("div");
    NativeRenderer::set_attr(&info, "data-flex", "1.0");
    NativeRenderer::set_attr(&info, "data-layout", "column");

    let n = NativeRenderer::text(name);
    NativeRenderer::set_attr(&n, "data-color", theme.text);
    NativeRenderer::set_attr(&n, "data-text-size", "16");
    NativeRenderer::set_attr(&n, "data-h", "30");
    NativeRenderer::append(&info, &n);

    let d = NativeRenderer::text(detail);
    NativeRenderer::set_attr(&d, "data-color", theme.muted);
    NativeRenderer::set_attr(&d, "data-text-size", "14");
    NativeRenderer::set_attr(&d, "data-h", "22");
    NativeRenderer::append(&info, &d);

    NativeRenderer::append(&row, &info);

    // Filled pill — color-coded so ON/OFF is scannable without reading text.
    let pill = NativeRenderer::element("div");
    NativeRenderer::set_attr(&pill, "data-w", "60");
    NativeRenderer::set_attr(&pill, "data-h", "54");
    NativeRenderer::set_attr(&pill, "data-fill", if enabled { "#173a20" } else { "#0e0e16" });

    let pill_txt = NativeRenderer::text(if enabled { "● ON" } else { "○ OFF" });
    NativeRenderer::set_attr(&pill_txt, "data-color", if enabled { "#44ff66" } else { "#3a3a50" });
    NativeRenderer::set_attr(&pill_txt, "data-text-size", "14");
    NativeRenderer::set_attr(&pill_txt, "data-text-weight", "bold");
    NativeRenderer::set_attr(&pill_txt, "data-h", "54");
    NativeRenderer::set_attr(&pill_txt, "data-align", "center");
    NativeRenderer::append(&pill, &pill_txt);
    NativeRenderer::append(&row, &pill);

    // fire_event has no bubbling — attach to every non-text child div that would
    // otherwise absorb the click silently, and to the row itself for edge areas.
    let mk = |a: Arc<Mutex<AutomationConfig>>, t: Arc<dyn Fn(&mut AutomationConfig) + Send + Sync>| {
        move |_: BrickEvent| { if let Ok(mut c) = a.lock() { t(&mut c); } }
    };
    NativeRenderer::on_event(&info, "click", Box::new(mk(Arc::clone(&auto_arc), Arc::clone(&toggle))));
    NativeRenderer::on_event(&pill, "click", Box::new(mk(Arc::clone(&auto_arc), Arc::clone(&toggle))));
    NativeRenderer::on_event(&row,  "click", Box::new(mk(auto_arc, toggle)));

    NativeRenderer::append(parent, &row);

    // Hairline separator
    let sep = NativeRenderer::element("div");
    NativeRenderer::set_attr(&sep, "data-h", "1");
    NativeRenderer::set_attr(&sep, "data-fill", theme.divider);
    NativeRenderer::append(parent, &sep);
}

/// Fixed-width button for the threshold segmented control (36×44 px).
fn thresh_btn(label: &str, theme: &Theme) -> NativeNode {
    let btn = NativeRenderer::element("div");
    NativeRenderer::set_attr(&btn, "data-fill", theme.surface_hi);
    NativeRenderer::set_attr(&btn, "data-w", "36");
    NativeRenderer::set_attr(&btn, "data-h", "44");

    let t = NativeRenderer::text(label);
    NativeRenderer::set_attr(&t, "data-color", theme.accent);
    NativeRenderer::set_attr(&t, "data-text-size", "20");
    NativeRenderer::set_attr(&t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&t, "data-h", "44");
    NativeRenderer::set_attr(&t, "data-align", "center");
    NativeRenderer::append(&btn, &t);

    btn
}

// ── Status tab ────────────────────────────────────────────────────────────────

fn status_tab(state: &ControlPanelState, theme: &Theme, parent: &NativeNode) {
    // ── System ────────────────────────────────────────────────────────────────
    section_header("SYSTEM STATUS", "Connection and overlay lifecycle.", theme, parent);

    let (game_txt, game_col) = if state.game_active {
        ("ACTIVE — overlay coaching in progress", theme.ok)
    } else {
        ("STANDBY — waiting for League of Legends", theme.muted)
    };
    status_row("Game detection", game_txt, game_col, theme, parent);
    status_row("LCU / rune write", "Connects on demand when League is open", theme.accent2, theme, parent);
    status_row("Minimap CV",       "5 Hz frame scan (Windows only)", theme.accent2, theme, parent);

    spacer(20, parent);

    // ── Coaching engine modules ───────────────────────────────────────────────
    section_header(
        "COACHING ENGINE",
        "Active detection modules — all fire cues into the in-game feed.",
        theme,
        parent,
    );

    coaching_module(
        "Objective timers",
        "Dragon · Baron · Herald",
        "Tracks spawn countdowns and fires group-up / ward cues ~30 s before each.",
        theme.ok, theme, parent,
    );
    coaching_module(
        "Wave management",
        "CrashWave · WaveAlert",
        "Minimap CV clusters ally/enemy minions by lane; tells you when to crash or defend.",
        theme.ok, theme, parent,
    );
    coaching_module(
        "Summoner spell tracker",
        "SummonerSpellDown",
        "Infers Flash / TP usage from kill events (~280 s Flash, 210 s TP window).",
        theme.ok, theme, parent,
    );
    coaching_module(
        "Power spike detection",
        "EnemyPowerSpike",
        "Watches the lane opponent's item count each frame; fires when they complete a new item.",
        theme.ok, theme, parent,
    );
    coaching_module(
        "Vision analysis",
        "VisionGap",
        "Compares objective spawn timers against active ward positions near each pit.",
        theme.ok, theme, parent,
    );
    coaching_module(
        "Jungler tracking",
        "JunglerSpotted · JunglerUnknown",
        "CV icon matching on minimap surfaces enemy jungler position; warns when off-screen.",
        theme.ok, theme, parent,
    );
    coaching_module(
        "Lane metrics",
        "CS · Level · Kill pressure",
        "Live scoreboard diffs: CS deficit, level-up spikes, kill lead / loss in lane.",
        theme.ok, theme, parent,
    );
}

fn coaching_module(
    name:    &str,
    cues:    &str,
    detail:  &str,
    active_color: &str,
    theme:   &Theme,
    parent:  &NativeNode,
) {
    let row = hrow(52);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    NativeRenderer::set_attr(&row, "data-fill", theme.surface);
    NativeRenderer::set_attr(&row, "data-border-left", &format!("{}:2", active_color));

    // Name + cue tags
    let left = NativeRenderer::element("div");
    NativeRenderer::set_attr(&left, "data-flex", "0.28");
    NativeRenderer::set_attr(&left, "data-layout", "column");

    let n = NativeRenderer::text(name);
    NativeRenderer::set_attr(&n, "data-color", theme.text);
    NativeRenderer::set_attr(&n, "data-text-size", "14");
    NativeRenderer::set_attr(&n, "data-text-weight", "bold");
    NativeRenderer::set_attr(&n, "data-h", "22");
    NativeRenderer::append(&left, &n);

    let c = NativeRenderer::text(cues);
    NativeRenderer::set_attr(&c, "data-color", active_color);
    NativeRenderer::set_attr(&c, "data-text-size", "14");
    NativeRenderer::set_attr(&c, "data-h", "20");
    NativeRenderer::append(&left, &c);

    NativeRenderer::append(&row, &left);

    // Detail description
    let d = NativeRenderer::text(detail);
    NativeRenderer::set_attr(&d, "data-color", theme.muted);
    NativeRenderer::set_attr(&d, "data-text-size", "14");
    NativeRenderer::set_attr(&d, "data-flex", "0.72");
    NativeRenderer::set_attr(&d, "data-h", "52");
    NativeRenderer::append(&row, &d);

    NativeRenderer::append(parent, &row);

    // Hairline separator
    let sep = NativeRenderer::element("div");
    NativeRenderer::set_attr(&sep, "data-h", "1");
    NativeRenderer::set_attr(&sep, "data-fill", theme.divider);
    NativeRenderer::append(parent, &sep);
}

fn status_row(label: &str, value: &str, value_color: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(32);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    NativeRenderer::set_attr(&row, "data-fill", theme.surface);

    let lbl = NativeRenderer::text(&format!("{label}: "));
    NativeRenderer::set_attr(&lbl, "data-color", theme.muted);
    NativeRenderer::set_attr(&lbl, "data-text-size", "14");
    NativeRenderer::set_attr(&lbl, "data-flex", "0.25");
    NativeRenderer::append(&row, &lbl);

    let val = NativeRenderer::text(value);
    NativeRenderer::set_attr(&val, "data-color", value_color);
    NativeRenderer::set_attr(&val, "data-text-size", "14");
    NativeRenderer::set_attr(&val, "data-flex", "0.75");
    NativeRenderer::append(&row, &val);

    NativeRenderer::append(parent, &row);
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Absolute-positioned panel.
fn abs_panel(x: i32, y: i32, w: u32, h: u32, fill: &str) -> NativeNode {
    let p = NativeRenderer::element("div");
    NativeRenderer::set_attr(&p, "data-overlay", "true");
    NativeRenderer::set_attr(&p, "data-x", &x.to_string());
    NativeRenderer::set_attr(&p, "data-y", &y.to_string());
    NativeRenderer::set_attr(&p, "data-w", &w.to_string());
    NativeRenderer::set_attr(&p, "data-h", &h.to_string());
    NativeRenderer::set_attr(&p, "data-fill", fill);
    p
}

/// Fixed-height horizontal row. Uses `data-height` so layout is correct in
/// both overlay-child and non-overlay column contexts.
fn hrow(h: u32) -> NativeNode {
    let row = NativeRenderer::element("div");
    NativeRenderer::set_attr(&row, "data-layout", "row");
    NativeRenderer::set_attr(&row, "data-height", &h.to_string());
    row
}

fn spacer(h: u32, parent: &NativeNode) {
    let gap = NativeRenderer::element("div");
    NativeRenderer::set_attr(&gap, "data-height", &h.to_string());
    NativeRenderer::append(parent, &gap);
}

fn section_header(title: &str, subtitle: &str, theme: &Theme, parent: &NativeNode) {
    let row = hrow(60);
    NativeRenderer::set_attr(&row, "data-pad", "0 0 0 20");
    NativeRenderer::set_attr(&row, "data-fill", theme.surface);
    NativeRenderer::set_attr(&row, "data-border-bottom", &format!("{}:1", theme.divider));

    let t = NativeRenderer::text(title);
    NativeRenderer::set_attr(&t, "data-color", theme.text);
    NativeRenderer::set_attr(&t, "data-text-size", "18");
    NativeRenderer::set_attr(&t, "data-text-weight", "bold");
    NativeRenderer::set_attr(&t, "data-flex", "0.30");
    NativeRenderer::set_attr(&t, "data-h", "60");
    NativeRenderer::append(&row, &t);

    let s = NativeRenderer::text(subtitle);
    NativeRenderer::set_attr(&s, "data-color", theme.muted);
    NativeRenderer::set_attr(&s, "data-text-size", "15");
    NativeRenderer::set_attr(&s, "data-flex", "0.70");
    NativeRenderer::set_attr(&s, "data-h", "60");
    NativeRenderer::append(&row, &s);

    NativeRenderer::append(parent, &row);

    let gap = NativeRenderer::element("div");
    NativeRenderer::set_attr(&gap, "data-h", "8");
    NativeRenderer::append(parent, &gap);
}

fn rune_path_name(id: u32) -> &'static str {
    match id {
        8000 => "Precision",
        8100 => "Domination",
        8200 => "Sorcery",
        8300 => "Inspiration",
        8400 => "Resolve",
        _    => "Unknown",
    }
}
