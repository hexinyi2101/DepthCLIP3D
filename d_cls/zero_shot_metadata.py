import os

import pandas as pd

OPENAI_IMAGENET_TEMPLATES = (
    lambda c: f'a bad depth photo of a {c}.',
    lambda c: f'a depth photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a depth photo of the hard to see {c}.',
    lambda c: f'a low resolution depth photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad depth photo of the {c}.',
    lambda c: f'a cropped depth photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a depth photo of a hard to see {c}.',
    lambda c: f'a bright depth photo of a {c}.',
    lambda c: f'a depth photo of a clean {c}.',
    lambda c: f'a depth photo of a dirty {c}.',
    lambda c: f'a dark depth photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a depth photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a depth photo of the cool {c}.',
    lambda c: f'a close-up depth photo of a {c}.',
    lambda c: f'a black and white depth photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated depth photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright depth photo of the {c}.',
    lambda c: f'a cropped depth photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a depth photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted depth photo of a {c}.',
    lambda c: f'a blurry depth photo of the {c}.',
    lambda c: f'a depth photo of the {c}.',
    lambda c: f'a good depth photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a depth photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up depth photo of the {c}.',
    lambda c: f'a depth photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution depth photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a depth photo of the clean {c}.',
    lambda c: f'a depth photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a depth photo of a nice {c}.',
    lambda c: f'a depth photo of a weird {c}.',
    lambda c: f'a blurry depth photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated depth photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted depth photo of the {c}.',
    lambda c: f'a good depth photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a depth photo of the nice {c}.',
    lambda c: f'a depth photo of the small {c}.',
    lambda c: f'a depth photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a depth photo of the large {c}.',
    lambda c: f'a black and white depth photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark depth photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a depth photo of a cool {c}.',
    lambda c: f'a depth photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
)


# a much smaller subset of above prompts
# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad depth photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a depth photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a depth photo of the small {c}.',
)


IMAGENET_CLASSNAMES = (

)


CLASSNAMES = {
    'NYUV2': (
        "bathroom", "bedroom", "bookstore", "classroom", "dining room",
        "home office", "kitchen", "living room", "office", "others"
    ),
    'SUNRGBD': (
        "bathroom", "bedroom", "classroom", "computer room", "conference room", "corridor", "dining area",
        "dining room", "discussion area", "furniture store", "home office", "kitchen", "lab", "lecture theatre",
        "library", "living room", "office", "rest space", "study space"
    ),
    'SUNRGBDMAT': (
        "bathroom", "bedroom", "classroom", "computer room", "conference room", "corridor", "dining area",
        "dining room", "discussion area", "furniture store", "home office", "kitchen", "lab", "lecture theatre",
        "library", "living room", "office", "rest space", "study space"
    ),
    'nyunew':("A bathroom, often equipped with a toilet, sink, and shower, is where people typically perform their personal hygiene routines", "A bedroom is a private room where people sleep and relax, often containing a bed and, sometimes, furniture like wardrobes and dressers", "A bookstore is a retail establishment that sells books, magazines, and sometimes additional items like stationery or gifts", "A classroom is a learning space where students are taught by a teacher, often found in schools and educational institutions", "The dining room is a designated area in a home where people gather to eat meals together around a large table", "A home office is a space within a residence set aside for work or study, often equipped with a desk, computer, and other necessary supplies", "A kitchen is a room used for cooking and food preparation, typically equipped with appliances like a stove, refrigerator, and sink", "The living room is a communal space in a home where family members gather to relax, entertain guests, and engage in leisure activities", "An office is a room or set of rooms where business, professional, or administrative work is conducted","others"),
    'twonet': (
        "shoe shop","shopping basket","bakery","balloon","balance beam"
    ),
    'shapenet12': (
        "bag","birdhouse","camera","computer keyboard","dishwashing machine","earphone","mailbox","microphone","pillow","remote control","rocket","skateboard"
    ),
    'net': (
        'gasmask', 'espresso', 'screw', 'goldfinch', 'matchstick', 'space bar', 'moped', 'beach wagon', 'violin', 'sundial', 'black-and-tan coonhound', 'eggnog', 'cannon', 'pillow', 'wreck', 'reflex camera', 'black-footed ferret', 'sweatshirt', 'night snake', 'cornet', 'pencil box', 'boxer', 'streetcar', 'forklift', 'sliding door', 'bookcase', 'earthstar', 'yurt', 'barbershop', 'cairn', 'wallet', 'cabbage butterfly', 'African grey', 'typewriter keyboard', 'hermit crab', 'Pekinese', 'trombone', 'llama', 'street sign', 'redbone', 'acorn', 'French loaf', 'flat-coated retriever', 'volleyball', 'jinrikisha', 'puffer', 'ocarina', 'space shuttle', 'dowitcher', 'bucket', 'torch', 'magnetic compass', 'zebra', 'hornbill', 'gibbon', 'crutch', 'modem', 'chiton', 'Newfoundland', 'vase', 'lesser panda', 'Christmas stocking', 'dome', 'can opener', 'mountain tent', 'bell pepper', 'parking meter', 'hoopskirt', 'library', 'coffeepot', 'wool', 'pug', 'damselfly', 'affenpinscher', 'scorpion', 'bluetick', 'slide rule', 'Border terrier', 'Sussex spaniel', 'Great Pyrenees', 'otter', 'Dandie Dinmont', 'quail', 'lakeside', 'barrow', 'Old English sheepdog', 'cardoon', 'sandal', 'French horn', 'convertible', 'Bouvier des Flandres', 'daisy', 'African hunting dog', 'iron', 'crane', 'Rottweiler', 'vulture', 'Siamese cat', 'bald eagle', 'acoustic guitar', 'Australian terrier', 'harvester', 'panpipe', 'long-horned beetle', 'Welsh springer spaniel', 'conch', 'book jacket', 'rifle', 'television', 'butternut squash', 'collie', 'lipstick', 'oxygen mask', 'golden retriever', 'killer whale', 'weevil', 'dining table', 'carton', 'giant schnauzer', 'meat loaf', 'pitcher', 'snowplow', 'hip', 'tarantula', 'golf ball', 'beer glass', 'cello', 'swab', 'bagel', 'Boston bull', 'barracouta', 'valley', 'hamster', 'limpkin', 'backpack', 'lemon', 'cowboy hat', 'malamute', 'obelisk', 'Norfolk terrier', 'mosque', 'trailer truck', 'Shih-Tzu', 'water snake', 'paintbrush', 'chest', 'airliner', 'power drill', 'hammerhead', 'barn', 'snowmobile', 'muzzle', 'crash helmet', 'shower curtain', 'spoonbill', 'Norwegian elkhound', 'lionfish', 'pencil sharpener', 'cassette player', 'whiskey jug', 'three-toed sloth', 'piggy bank', 'agama', 'wine bottle', 'hare', 'parachute', 'entertainment center', 'jay', 'lens cap', 'jellyfish', 'pizza', 'window screen', 'warthog', 'turnstile', 'beagle', 'African crocodile', 'European fire salamander', 'racket', 'revolver', 'standard schnauzer', 'paddle', 'albatross', 'planetarium', 'Great Dane', 'Appenzeller', 'barrel', 'Irish water spaniel', 'red wine', 'malinois', 'electric locomotive', 'cliff dwelling', 'kelpie', 'radiator', 'macaque', 'velvet', 'siamang', 'mantis', 'red-backed sandpiper', 'boathouse', 'lumbermill', 'Yorkshire terrier', 'overskirt', 'digital clock', 'ladle', 'strawberry', 'American egret', 'sorrel', 'fox squirrel', 'toyshop', 'tiger beetle', 'guenon', 'chow', 'sloth bear', 'suspension bridge', 'fire screen', 'snow leopard', 'bison', 'red fox', 'dung beetle', 'mitten', 'thatch', 'Saint Bernard', 'brass', 'handkerchief', 'croquet ball', 'water tower', 'aircraft carrier', 'jaguar', 'skunk', 'oxcart', 'frilled lizard', 'bustard', 'Eskimo dog', 'plastic bag', 'military uniform', 'dial telephone', 'mushroom', 'tiger cat', 'pajama', 'whistle', 'pill bottle', 'plow', 'junco', 'snorkel', 'tabby', 'green snake', 'orangutan', 'loggerhead', 'crib', 'toilet seat', 'alp', 'snail', 'bathing cap', 'cup', 'assault rifle', 'washbasin', 'crossword puzzle', 'beaker', 'gar', 'chambered nautilus', 'Italian greyhound', 'wall clock', 'cockroach', 'throne', 'vacuum', 'electric ray', 'pick', 'wing', 'ptarmigan', 'fiddler crab', 'gondola', 'ruddy turnstone', 'Ibizan hound', 'web site', 'crayfish', 'tench', 'trifle', 'baseball', 'bolo tie', 'carbonara', 'platypus', 'upright', 'schooner', 'oystercatcher', 'cauliflower', 'patas', 'chocolate sauce', 'sea lion', 'packet', 'bassinet', 'Irish terrier', 'microwave', 'rhinoceros beetle', 'hourglass', 'folding chair', 'guinea pig', 'coho', 'nipple', 'bulbul', 'stove', 'dugong', 'fly', 'tusker', 'desktop computer', 'terrapin', 'garbage truck', 'stinkhorn', 'Siberian husky', 'Bedlington terrier', 'American lobster', 'fur coat', 'American alligator', 'china cabinet', 'house finch', 'chickadee', 'buckeye', 'printer', 'tray', 'promontory', 'alligator lizard', 'shopping cart', 'corkscrew', 'rock python', 'trimaran', 'proboscis monkey', "potter's wheel", 'feather boa', 'limousine', 'marmoset', 'pickelhaube', 'briard', 'lab coat', 'komondor', 'wire-haired fox terrier', 'remote control', 'totem pole', 'greenhouse', 'chimpanzee', 'football helmet', 'coil', 'sea anemone', 'whiptail', 'tow truck', 'cassette', 'wallaby', 'otterhound', 'German short-haired pointer', 'ping-pong ball', 'sidewinder', 'leopard', 'American black bear', 'photocopier', 'goldfish', 'hook', 'seashore', 'bookshop', 'cocktail shaker', 'pool table', 'eel', 'sea slug', 'traffic light', 'necklace', 'stupa', 'drum', 'maillot', 'combination lock', 'accordion', 'mosquito net', 'triceratops', 'hummingbird', 'great white shark', 'cricket', 'standard poodle', 'centipede', 'ram', 'monitor', 'ambulance', 'hognose snake', 'ear', 'tank', 'parallel bars', 'chainlink fence', 'lynx', 'bolete', 'Border collie', 'isopod', 'hyena', 'ostrich', 'sea urchin', 'missile', 'spider web', 'digital watch', 'burrito', 'flamingo', 'mousetrap', 'unicycle', 'goblet', 'English setter', 'mask', 'half track', 'Rhodesian ridgeback', 'harmonica', 'rock beauty', 'king crab', 'hotdog', 'wolf spider', 'ringneck snake', 'puck', 'apiary', 'axolotl', 'computer keyboard', 'sombrero', 'frying pan', "jack-o'-lantern", 'Leonberg', 'cheetah', 'ice cream', 'teddy', 'police van', 'sock', 'agaric', 'Angora', 'cuirass', 'hard disc', 'Granny Smith', 'giant panda', 'refrigerator', 'starfish', 'toy poodle', 'radio telescope', 'coyote', 'indri', 'curly-coated retriever', 'scuba diver', 'airship', 'studio couch', 'electric fan', 'drilling platform', 'grand piano', 'kuvasz', 'langur', 'garter snake', 'African chameleon', 'joystick', 'pineapple', 'tiger', 'tailed frog', 'soap dispenser', 'horned viper', 'hamper', 'spiny lobster', 'gas pump', 'Afghan hound', 'reel', 'magpie', 'dishrag', 'cab', 'borzoi', 'shoe shop', 'go-kart', 'black grouse', 'pinwheel', 'notebook', 'jeep', 'goose', 'cock', 'birdhouse', 'gong', 'rain barrel', 'mashed potato', 'Cardigan', 'laptop', 'steam locomotive', 'white stork', 'loudspeaker', 'submarine', 'fireboat', 'hair spray', 'kit fox', 'EntleBucher', 'Brabancon griffon', 'seat belt', 'spaghetti squash', 'tricycle', 'Sealyham terrier', 'rock crab', 'sulphur butterfly', 'wok', 'Madagascar cat', 'bath towel', 'viaduct', 'partridge', 'cowboy boot', 'spider monkey', 'chiffonier', 'hog', 'apron', 'doormat', 'miniskirt', 'thunder snake', 'black and gold garden spider', 'Chihuahua', 'radio', 'artichoke', 'wig', 'plate rack', 'plunger', 'miniature poodle', 'ski', 'European gallinule', 'loupe', 'Polaroid camera', 'pretzel', 'candle', 'keeshond', 'lifeboat', 'microphone', 'electric guitar', 'washer', 'stethoscope', 'safety pin', 'nail', 'ant', 'motor scooter', 'admiral', 'bathtub', 'dam', 'grey fox', 'Kerry blue terrier', 'swing', 'breakwater', 'Scotch terrier', 'marimba', 'American chameleon', 'polecat', 'sarong', 'English springer', 'rotisserie', 'pomegranate', 'pirate', 'barn spider', 'peacock', 'Norwich terrier', 'horse cart', 'brambling', 'shower cap', 'badger', 'great grey owl', 'ski mask', 'jean', 'mailbox', 'castle', 'grocery store', 'beacon', 'soccer ball', 'bicycle-built-for-two', 'horizontal bar', 'German shepherd', 'crate', 'dhole', 'coral fungus', 'breastplate', 'tape player', 'groom', 'howler monkey', 'Arctic fox', 'Greater Swiss Mountain dog', 'custard apple', 'manhole cover', 'Staffordshire bullterrier', 'bakery', 'water bottle', 'restaurant', 'Crock Pot', 'running shoe', 'pot', 'picket fence', 'acorn squash', 'toy terrier', 'geyser', 'shoji', 'hammer', 'gazelle', 'sea snake', 'golfcart', 'home theater', 'black swan', 'umbrella', 'flute', 'wood rabbit', 'Weimaraner', 'container ship', 'table lamp', 'red wolf', 'balance beam', 'green lizard', 'rule', 'chime', 'wardrobe', 'Indian elephant', 'Brittany spaniel', 'leatherback turtle', 'barbell', 'tobacco shop', 'buckle', 'West Highland white terrier', 'screen', 'stole', 'jigsaw puzzle', 'fountain', 'sports car', 'leafhopper', 'yawl', 'Egyptian cat', 'baboon', 'sea cucumber', 'measuring cup', 'barber chair', 'Pomeranian', 'hartebeest', 'hen', 'spotlight', 'ringlet', 'balloon', 'consomme', 'steel drum', 'dalmatian', 'park bench', 'speedboat', 'mongoose', 'Irish wolfhound', 'clog', 'binder', 'maypole', 'mortar', 'pickup', 'Arabian camel', 'perfume', 'stingray', 'suit', 'dragonfly', 'honeycomb', 'cinema', 'shield', 'sunscreen', 'hatchet', 'plate', 'bannister', 'armadillo', 'cliff', 'grille', 'chain', 'sax', 'vestment', 'clumber', 'shovel', 'mixing bowl', 'lycaenid', 'eft', 'tennis ball', 'thimble', 'steel arch bridge', 'brain coral', 'mink', 'bull mastiff', 'passenger car', 'head cabbage', 'warplane', 'harvestman', 'Gordon setter', 'toilet tissue', 'cocker spaniel', 'coral reef', 'abacus', 'bighorn', 'diamondback', 'oil filter', 'stretcher', 'toaster', 'maze', 'lawn mower', 'porcupine', 'king penguin', 'palace', 'academic gown', 'file', 'monarch', 'American Staffordshire terrier', 'timber wolf', 'Indian cobra', 'water jug', 'prayer rug', 'car mirror', 'ballplayer', 'water buffalo', 'rocking chair', 'volcano', 'chain mail', 'bulletproof vest', 'rubber eraser', 'projector', 'butcher shop', 'black widow', 'cradle', 'Shetland sheepdog', 'solar dish', 'gyromitra', 'comic book', 'sunglass', 'fountain pen', 'drumstick', 'fig', 'jersey', 'lorikeet', 'cicada', 'punching bag', 'miniature schnauzer', 'banana', 'mobile home', 'brassiere', 'Scottish deerhound', 'titi', 'strainer', 'toucan', 'bee', 'anemone fish', 'lotion', 'basset', 'scabbard', 'impala', 'flagpole', 'beer bottle', 'bikini', 'coffee mug', 'zucchini', 'abaya', 'bow', 'water ouzel', 'jackfruit', 'carousel', 'capuchin', 'neck brace', 'potpie', 'tree frog', 'tractor', 'kimono', "yellow lady's slipper", 'iPod', 'Dungeness crab', 'drake', 'analog clock', 'African elephant', 'king snake', 'boa constrictor', 'garden spider', 'Doberman', 'ruffed grouse', 'purse', 'organ', 'syringe', 'swimming trunks', 'mortarboard', 'sewing machine', 'catamaran', 'leaf beetle', 'Petri dish', 'hay', 'Persian cat', 'chain saw', 'diaper', 'padlock', 'saltshaker', 'poncho', 'basenji', 'bow tie', 'disk brake', 'weasel', 'sunglasses', 'grasshopper', 'trilobite', 'tile roof', 'slug', 'teapot', 'American coot', 'knot', 'vault', 'mailbag', 'confectionery', 'sleeping bag', 'hair slide', 'bell cote', 'lion', 'white wolf', 'school bus', 'fire engine', 'caldron', 'window shade', 'lighter', 'espresso maker', 'silky terrier', 'gorilla', 'stopwatch', 'pole', 'lacewing', 'spindle', 'cleaver', 'medicine chest', 'kite', 'harp', 'Windsor tie', 'mountain bike', 'lampshade', 'sulphur-crested cockatoo', 'recreational vehicle', 'bottlecap', 'ground beetle', 'marmot', 'maraca', 'tub', 'sandbar', 'stone wall', 'hand-held computer', 'macaw', 'bassoon', 'Labrador retriever', 'bearskin', 'ibex', 'switch', 'scale', 'projectile', 'megalith', 'quill', 'knee pad', 'ashcan', 'whippet', 'English foxhound', 'bee eater', 'menu', 'tick', 'Pembroke', 'monastery', 'bullfrog', 'plane', 'mud turtle', 'letter opener', 'vending machine', 'ballpoint', 'ox', 'squirrel monkey', 'red-breasted merganser', 'broccoli', 'ladybug', 'bullet train', 'pop bottle', 'freight car', 'tripod', 'pay-phone', 'walking stick', 'Lhasa', 'trolleybus', 'bobsled', 'hot pot', 'cardigan', 'redshank', 'dogsled', 'miniature pinscher', 'papillon', 'dough', 'bloodhound', 'four-poster', 'echidna', 'wooden spoon', 'dishwasher', 'beaver', 'scoreboard', 'gown', 'grey whale', 'coucal', 'Dutch oven', 'basketball', 'French bulldog', 'spatula', 'cellular telephone', 'safe', 'brown bear', 'patio', 'soup bowl', 'common iguana', 'cucumber', 'Band Aid', 'cloak', 'vine snake', 'envelope', 'jacamar', 'theater curtain', 'Chesapeake Bay retriever', 'liner', 'bittern', 'barometer', 'odometer', 'Maltese dog', 'Saluki', 'mouse', 'Samoyed', 'paper towel', 'oscilloscope', 'dumbbell', 'paddlewheel', 'black stork', 'vizsla', 'ice bear', 'minivan', 'racer', 'rugby ball', 'groenendael', 'Loafer', 'Airedale', 'Blenheim spaniel', 'Tibetan mastiff', 'prison', 'canoe', 'Bernese mountain dog', 'wombat', 'banded gecko', 'hippopotamus', 'bib', 'soft-coated wheaten terrier', 'prairie chicken', 'Mexican hairless', 'flatworm', 'pelican', 'robin', 'hen-of-the-woods', 'binoculars', 'Model T', 'cash machine', 'corn', 'oboe', 'Irish setter', 'bubble', 'broom', 'guacamole', 'cougar', 'dingo', 'rapeseed', 'Tibetan terrier', 'moving van', 'indigo bunting', 'common newt', 'spotted salamander', 'banjo', 'triumphal arch', 'altar', 'minibus', 'milk can', 'nematode', 'colobus', "carpenter's kit", 'desk', 'little blue heron', 'thresher', 'trench coat', 'schipperke', 'bonnet', 'meerkat', 'space heater', 'screwdriver', 'Japanese spaniel', 'amphibian', 'sturgeon', 'Gila monster', 'Lakeland terrier', 'guillotine', 'orange', 'waffle iron', 'stage', 'ice lolly', 'wild boar', 'slot', 'green mamba', 'dock', 'Walker hound', 'pedestal', 'worm fence', 'Komodo dragon', 'car wheel', 'holster', 'koala', 'quilt', 'cheeseburger', 'face powder', 'box turtle', 'hand blower', 'church', 'tiger shark', 'shopping basket', 'pier', 'CD player'
    )
}
