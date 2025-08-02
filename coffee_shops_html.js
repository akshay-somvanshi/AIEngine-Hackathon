const coffeeShops = [
    {
        name: "Monmouth Coffee Company",
        address: "27 Monmouth Street, Covent Garden",
        lat: 51.5129,
        lng: -0.127,
        coffees: [
            {
                name: "El Socorro Maracaturra",
                origin: "Guatemala",
            },
            {
                name: "Washed Kenya",
                origin: "South-Central Kenya",
            },
            {
                name: "Colombia Tres Dragones Natural",
                origin: "Valle De Cauca Department",
            },
            {
                name: "Ethiopia Yirgacheffe Kochere Debo",
                origin: "Southern Ethiopia",
            },
            {
                name: "Rwanda Nyamagabe Kigeme Lot #4",
                origin: "Southern Province",
            },
            {
                name: "Kona Geisha Champagne Natural",
                origin: "Kona",
            },
            {
                name: "Indonesia Emerald Mandheling",
                origin: "Aceh Province",
            },
            {
                name: "El Salvador",
                origin: "Northwest El Salvador",
            },
            {
                name: "Josh Daniels Grateful Blend",
                origin: "Colombia",
            },
            {
                name: "Kenya Kamwangi",
                origin: "Kirinyaga County",
            },
        ]
    },
    {
        name: "Monmouth Coffee Company",
        address: "2 Park Street, Borough Market",
        lat: 51.503,
        lng: -0.094,
        coffees: [
            {
                name: "Ethiopia Natural Yirgacheffe Adado Feysa Dira",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Sidama Extreme Papilio Natural",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Kenya Mwiria",
                origin: "Sidama (Also Sidamo) Growing Region",
            },
            {
                name: "Burundi Nemba",
                origin: "Kabuye",
            },
            {
                name: "Three Queens Blend",
                origin: "Burundi",
            },
            {
                name: "El Salvador Malacara B Orange",
                origin: "Volcán Santa Ana",
            },
            {
                name: "Ethiopia Guji Hambela Buku Saysa Natural Process",
                origin: "Oromia Region",
            },
        ]
    },
    {
        name: "Monmouth Coffee Company",
        address: "27 Maltby Street, Bermondsey",
        lat: 51.498,
        lng: -0.08,
        coffees: [
            {
                name: "Hacienda La Esmeralda Buenos Aires Geisha Natural",
                origin: "Panama",
            },
            {
                name: "Ethiopia Kochere G1",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia Kayon Mountain",
                origin: "Guji Zone",
            },
            {
                name: "Shakiso Natural",
                origin: "Oromia Region",
            },
            {
                name: "KonAroma Direct Trade Kona",
                origin: "Big Island Of Hawaii",
            },
            {
                name: "Black Level Blend",
                origin: "Colombia",
            },
            {
                name: "Primavera Colombia Espresso",
                origin: "Colombia",
            },
            {
                name: "Panama Don Julian Pacamara",
                origin: "Chiriqui Province",
            },
            {
                name: "Canoe Blend",
                origin: "South America",
            },
            {
                name: "El Salvador Anarquia Pacamara",
                origin: "El Salvador",
            },
        ]
    },
    {
        name: "Prufrock Coffee",
        address: "23-25 Leather Lane, Holborn",
        lat: 51.5189,
        lng: -0.1095,
        coffees: [
            {
                name: "Castillo El Paraiso",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Raro Boda",
                origin: "Oromia Region",
            },
            {
                name: "Finca La Aurora Camilina Geisha",
                origin: "Chiriqui Province",
            },
            {
                name: "Peru Yanesha Geisha",
                origin: "Villa Rica",
            },
            {
                name: "Ka‘ū Lactic Natural",
                origin: "Big Island Of Hawai‘I",
            },
            {
                name: "Peru Chacra Don Dago",
                origin: "Oxampampa",
            },
            {
                name: "Kitten Q Espresso Blend",
                origin: "Nicaragua",
            },
            {
                name: "Sumatra Lintong",
                origin: "North Sumatra Province",
            },
            {
                name: "Ethiopia Hambela Benti G1 Washed",
                origin: "Oromia Region",
            },
            {
                name: "Kenya Washed Nyeri Gichathaini Factory AB",
                origin: "South-Central Kenya",
            },
        ]
    },
    {
        name: "Workshop Coffee",
        address: "27 Clerkenwell Road, Clerkenwell",
        lat: 51.522,
        lng: -0.106,
        coffees: [
            {
                name: "Brazil Ipanema Black Edition A-41 Red Cherry",
                origin: "Brazil",
            },
            {
                name: "Kenya Gatuyaini",
                origin: "Bench-Maji Zone",
            },
            {
                name: "Yemen Al-Obbarat",
                origin: "Yemen",
            },
            {
                name: "Bait Alal Community",
                origin: "Sana’A Governorate",
            },
            {
                name: "Ethiopia Lecho Torka",
                origin: "Oromia Region",
            },
            {
                name: "Papua New Guinea Baroida Honey",
                origin: "Eastern Highlands Province",
            },
            {
                name: "Bean Series Ethiopia Guji Raro",
                origin: "Oromia Region",
            },
            {
                name: "Floral Samurai",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Workshop Coffee",
        address: "80A Mortimer Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Private Reserve Rwanda Huye",
                origin: "Southern Rwanda",
            },
            {
                name: "Kibingo Burundi",
                origin: "Northern Burundi",
            },
            {
                name: "Colombia Tolima Monteverde Estate Natural Wush Wush",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Dimtu",
                origin: "Guji Zone",
            },
            {
                name: "Finca Sophia Gesha Washed",
                origin: "Chiriquí",
            },
        ]
    },
    {
        name: "Workshop Coffee",
        address: "75 Cowcross Street, Farringdon",
        lat: 51.522,
        lng: -0.106,
        coffees: [
            {
                name: "Lugmapata Ecuador",
                origin: "Chimborazo Province",
            },
            {
                name: "Kenya Kabare Konyu",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Sheka Kawo Kamina Natural",
                origin: "Oromia Region",
            },
            {
                name: "Thung Chang Robusta Honey Espresso",
                origin: "Thailand",
            },
            {
                name: "Kenya Kariru PB",
                origin: "South-Central Kenya",
            },
        ]
    },
    {
        name: "Ozone Coffee Roasters",
        address: "11 Leonard Street, Shoreditch",
        lat: 51.523,
        lng: -0.081,
        coffees: [
            {
                name: "The Aurora Project",
                origin: "Guatemala",
            },
            {
                name: "Berg Wu Championship Sidamo Washed G1 Lot 20-02",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia Kayon Mountain (Natural)",
                origin: "Guji Zone",
            },
            {
                name: "Organic Sumatra",
                origin: "Sumatra",
            },
            {
                name: "Full Moon Espresso",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Caravan Coffee Roasters",
        address: "11-13 Exmouth Market, Clerkenwell",
        lat: 51.523,
        lng: -0.106,
        coffees: [
            {
                name: "Kona Geisha Champagne Natural",
                origin: "Kona",
            },
            {
                name: "Kenya Mwiria",
                origin: "Sidama (Also Sidamo) Growing Region",
            },
            {
                name: "Gelena Geisha",
                origin: "Oromia Region",
            },
            {
                name: "Kona Mocca®",
                origin: "North Kona Growing District",
            },
            {
                name: "Sumatra Aceh Gayo Mountain Kenawat Raisin Honey",
                origin: "Aceh Province",
            },
            {
                name: "Burundi",
                origin: "Narino Department",
            },
        ]
    },
    {
        name: "Caravan Coffee Roasters",
        address: "152-156 Clerkenwell Road, Clerkenwell",
        lat: 51.522,
        lng: -0.106,
        coffees: [
            {
                name: "Feku Double",
                origin: "Oromia Region",
            },
            {
                name: "Chiayi Blend",
                origin: "Brazil",
            },
            {
                name: "El Salvador Finca Himalaya Pacamara",
                origin: "Ahuachapán Department",
            },
            {
                name: "Rwanda Rulindo Tumba",
                origin: "Rwanda",
            },
            {
                name: "Kibingo Burundi",
                origin: "Northern Burundi",
            },
            {
                name: "Trilogy",
                origin: "Africa",
            },
            {
                name: "El Salvador Santa Elena Pacamara",
                origin: "El Salvador",
            },
            {
                name: "Holiday Blend",
                origin: "Kenya",
            },
            {
                name: "Cuxinales Guatemala",
                origin: "Guatemala",
            },
            {
                name: "Costa Coast Blend",
                origin: "Ethiopia",
            },
        ]
    },
    {
        name: "Allpress Espresso",
        address: "58 Redchurch Street, Shoreditch",
        lat: 51.523,
        lng: -0.075,
        coffees: [
            {
                name: "Kenya AA Top Gatomboya",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Sidamo Bensa Asefa Dukamo Washed",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Buna Boka #1",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Finca Santa Isabel",
                origin: "Alta Verapaz",
            },
            {
                name: "Sidamo Suke Quto",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia Natural Hambela Wate",
                origin: "Oromia Region",
            },
            {
                name: "Rider Blend",
                origin: "Colombia",
            },
            {
                name: "Colombia Finca Lord Baltimore Papayo",
                origin: "Huila Department",
            },
        ]
    },
    {
        name: "Allpress Espresso",
        address: "55 Dalston Lane, Dalston",
        lat: 51.548,
        lng: -0.075,
        coffees: [
            {
                name: "Hacienda La Esmeralda Cabana Geisha Natural",
                origin: "Western Panama",
            },
            {
                name: "Jamaica Blue Mountain",
                origin: "Eastern Jamaica.",
            },
            {
                name: "Panama Finca Hartmann Geisha Natural",
                origin: "Far Western Panama",
            },
            {
                name: "Congo Kivu",
                origin: "South Kivu Province",
            },
            {
                name: "Coffea Diversa Bourbon Rey Guatemala",
                origin: "Jamaica",
            },
            {
                name: "Rwanda Sholi Natural Single-Origin Espresso",
                origin: "Rwanda",
            },
        ]
    },
    {
        name: "Kaffeine",
        address: "66 Great Titchfield Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Ka’u Classic Dark",
                origin: "Big Island Of Hawai’I",
            },
            {
                name: "Foothills Series Hambela Ethiopia",
                origin: "Guji Zone",
            },
            {
                name: "Kebele Village Espresso",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ecuador Loja Clara Sidra Natural Taza Dorada #2",
                origin: "Southern Ecuador",
            },
            {
                name: "Sumatra Yellow Bourbon",
                origin: "Simalungun",
            },
        ]
    },
    {
        name: "Kaffeine",
        address: "15 Eastcastle Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Thung Chang Robusta Honey Espresso",
                origin: "Thailand",
            },
            {
                name: "Honey Honduras Comsa",
                origin: "La Paz Department",
            },
            {
                name: "Typica Natural",
                origin: "North Kona Growing District",
            },
            {
                name: "Meridiano Ecuador",
                origin: "Pichincha Province",
            },
            {
                name: "British Style Espresso Blend",
                origin: "Honduras",
            },
            {
                name: "El Salvador Finca El Cerro Natural",
                origin: "El Salvador",
            },
            {
                name: "Dead Reckoning",
                origin: "El Salvador",
            },
            {
                name: "Ethiopia Limu",
                origin: "Oromia Region",
            },
            {
                name: "Kenya Karindudu AA",
                origin: "South-Central Kenya",
            },
            {
                name: "El Floral Colombia",
                origin: "Tolima Department",
            },
        ]
    },
    {
        name: "Notes Coffee",
        address: "31 St Martin's Lane, Covent Garden",
        lat: 51.512,
        lng: -0.128,
        coffees: [
            {
                name: "Ethiopia Sidamo Euphora Special Lot",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Colombia Antioqiua",
                origin: "Colombia",
            },
            {
                name: "Kenya AA Ares Phoenix Special",
                origin: "South-Central Kenya",
            },
            {
                name: "Finca El Potrero",
                origin: "Guatemala",
            },
            {
                name: "Colombia Finca El Paraiso Double Anaerobic Geisha",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Guji Buku",
                origin: "Guji Zone",
            },
        ]
    },
    {
        name: "Notes Coffee",
        address: "36 Trafalgar Square, Charing Cross",
        lat: 51.507,
        lng: -0.128,
        coffees: [
            {
                name: "Panama Baru Geisha Black",
                origin: "Western Panama",
            },
            {
                name: "No. 6 Espresso Blend",
                origin: "Brazil",
            },
            {
                name: "Ethiopia Yirgacheffe Awassa",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Ethiopia Yirgacheffe Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "La Belleza Colombia",
                origin: "Huila",
            },
            {
                name: "Laura’s Reserve SL-28",
                origin: "North Kona Growing District",
            },
            {
                name: "Thailand Robusta Panid Choosit",
                origin: "Chumphon",
            },
            {
                name: "Colombia Pink Bourbon Filadelfia",
                origin: "Colombia",
            },
            {
                name: "Galapagos La Tortuga",
                origin: "Galapagos Islands",
            },
        ]
    },
    {
        name: "Notes Coffee",
        address: "1 New Street, Covent Garden",
        lat: 51.512,
        lng: -0.128,
        coffees: [
            {
                name: "Colombia El Paraiso Floral Lychee",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Dame Dabaye",
                origin: "Oromia Region",
            },
            {
                name: "Ethiopia Guji Uraga Tebe Burka Natural G1",
                origin: "Oromia Region",
            },
            {
                name: "Burundi Kinyovu",
                origin: "Burundi",
            },
            {
                name: "Ethiopia Washed Shantawene",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Hunkute by Nordic Approach",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia Guji Natural Water-Processed Decaf",
                origin: "Oromia Region",
            },
            {
                name: "Signature Hazelnut Dark Chocolate",
                origin: "Guatemala",
            },
            {
                name: "Ethiopia Natural Phoenix Special “Andromeda” Espresso",
                origin: "Southwest Ethiopia",
            },
        ]
    },
    {
        name: "Flat White",
        address: "17 Berwick Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Panama Auromar Estate Geisha Peaberry",
                origin: "Chiriqui Province",
            },
            {
                name: "Moonbow Decaf Espresso",
                origin: "Caldas Department",
            },
            {
                name: "Karen J Kona Red Bourbon",
                origin: "North Kona Growing District",
            },
            {
                name: "Andi Sumatra",
                origin: "Sumatra",
            },
            {
                name: "Ethiopian Guji Ana Sora Alcoholic Natural G1",
                origin: "Southern Ethiopia",
            },
        ]
    },
    {
        name: "Flat White",
        address: "25 Frith Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Auromar Geisha Natural",
                origin: "Chiriqui Province",
            },
            {
                name: "Espresso Classico",
                origin: "Ecuador",
            },
            {
                name: "Colombia Geisha",
                origin: "Colombia",
            },
            {
                name: "Costa Rica Termico",
                origin: "Costa Rica",
            },
            {
                name: "Organic Ethiopia Kirite",
                origin: "Guji Zone",
            },
            {
                name: "Guatemala Coffea Diversa Geisha Queen Honey",
                origin: "Guatemala",
            },
            {
                name: "Ethiopia Natural Guji Sweet Lady",
                origin: "Oromia Region",
            },
        ]
    },
    {
        name: "The Coffee Works",
        address: "40-42 Great Titchfield Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Ethiopian Sweet Lily",
                origin: "Oromia Region",
            },
            {
                name: "Colombia El Paraiso Floral Lychee",
                origin: "Colombia",
            },
            {
                name: "Colombia Finca El Caucho Pink Bourbon",
                origin: "Colombia",
            },
            {
                name: "Baby Dragons",
                origin: "Valle De Cauca Department",
            },
            {
                name: "Geisha",
                origin: "“Big Island” Of Hawai’I",
            },
            {
                name: "Aged Sumatra Espresso",
                origin: "Huila",
            },
            {
                name: "Guatemala Washed Finca El General Pache",
                origin: "Guatemala",
            },
            {
                name: "Hawai’i Kilauea Volcano Yeast Fermentation Washed",
                origin: "Big Island Of Hawai’I",
            },
        ]
    },
    {
        name: "Department of Coffee and Social Affairs",
        address: "14-16 Leather Lane, Holborn",
        lat: 51.5189,
        lng: -0.1095,
        coffees: [
            {
                name: "ASOPCAFA Espresso",
                origin: "Southern Colombia",
            },
            {
                name: "Ethiopia Yirgacheffe Adado",
                origin: "Southern Ethiopia",
            },
            {
                name: "Maui Kupa’a Orange Bourbon",
                origin: "Island Of Maui",
            },
            {
                name: "Ardent Ethiopia Natural",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Kenya Nyeri Othaya Gatugi Red Cherry Project AA",
                origin: "South-Central Kenya",
            },
        ]
    },
    {
        name: "Department of Coffee and Social Affairs",
        address: "15-17 Old Street, Shoreditch",
        lat: 51.523,
        lng: -0.081,
        coffees: [
            {
                name: "Kuta Kofi Papua New Guinea",
                origin: "Jiawaka Province",
            },
            {
                name: "Ethiopia Guji Shakiso Dambi Udo Natural Anaerobic",
                origin: "Southern Ethiopia",
            },
            {
                name: "Kenya AB Muchoki",
                origin: "South-Central Kenya",
            },
            {
                name: "Guatemala Buena Vista Single-Origin Espresso",
                origin: "Guatemala",
            },
            {
                name: "Papua New Guinea “Papa Kinne”",
                origin: "Eastern Highlands",
            },
            {
                name: "Fruit Bomb/Ethiopia Shantewene",
                origin: "South-Central Ethiopia",
            },
        ]
    },
    {
        name: "TAP Coffee",
        address: "193 Wardour Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Costa Rica Aris Red Honey Lot 2002 Espresso",
                origin: "Costa Rica",
            },
            {
                name: "Cocaine for Coffee",
                origin: "Tolima Department",
            },
            {
                name: "Thailand Huai Mae Liam White Honey",
                origin: "Chiang Rai",
            },
            {
                name: "Butare Huye of Rwanda",
                origin: "Dolores",
            },
            {
                name: "Naturals With Attitude",
                origin: "Ethiopia",
            },
            {
                name: "Ethiopia Guji Shakisso Natural Lot 24",
                origin: "Oromia Region",
            },
            {
                name: "Esmeralda Estate Panama Geisha",
                origin: "Western Panama",
            },
            {
                name: "Elida Estate ASD Natural Catuai 15",
                origin: "Western Panama",
            },
        ]
    },
    {
        name: "TAP Coffee",
        address: "114 Tottenham Court Road, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Ecuador COE 1st place Arashi Typica Mejorado Washed",
                origin: "Loja",
            },
            {
                name: "Ethiopia Oromia Guji Washed",
                origin: "Oromia Region",
            },
            {
                name: "Yemen Al-Obbarat",
                origin: "Yemen",
            },
            {
                name: "Mexico Ozolotepec",
                origin: "Mexico",
            },
            {
                name: "Guatemala (from bulk bin)",
                origin: "Sidamo (Also Sidama) Growing Region",
            },
            {
                name: "Ethiopia Yirgacheffe Kochere",
                origin: "Southern Ethiopia",
            },
        ]
    },
    {
        name: "The Gentlemen Baristas",
        address: "63 Union Street, Borough",
        lat: 51.503,
        lng: -0.094,
        coffees: [
            {
                name: "Thiriku AA Kenya",
                origin: "Sidama (Also Sidamo) Growing Region",
            },
            {
                name: "Guatemala Buena Vista Single-Origin Espresso",
                origin: "Guatemala",
            },
            {
                name: "Congo Muungano",
                origin: "Democratic Republic Of The Congo",
            },
            {
                name: "Costa Rica Las Lajas Red Honey",
                origin: "Costa Rica",
            },
            {
                name: "La Esperanza Colombian Natural X.O.",
                origin: "Colombia",
            },
        ]
    },
    {
        name: "The Gentlemen Baristas",
        address: "44-46 Commercial Street, Spitalfields",
        lat: 51.52,
        lng: -0.075,
        coffees: [
            {
                name: "Rwanda Remera Kabeza",
                origin: "South-Central Rwanda",
            },
            {
                name: "Camilina Geisha",
                origin: "Chiriqui Province",
            },
            {
                name: "Aponte Colombia",
                origin: "Southern Colombia",
            },
            {
                name: "Ethiopia Nansebo Worka",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia Sidamo",
                origin: "Southern Ethiopia",
            },
            {
                name: "Yemen Haraaz Red Mahal Aqeeq ul Station Natural",
                origin: "Haraaz",
            },
            {
                name: "Ecuador Finca Lugmapata Mejorado",
                origin: "Chimborazo",
            },
            {
                name: "Nicaragua La Huella Catuaí",
                origin: "Nicaragua",
            },
        ]
    },
    {
        name: "WatchHouse Coffee",
        address: "8-10 Bermondsey Street, Bermondsey",
        lat: 51.498,
        lng: -0.08,
        coffees: [
            {
                name: "Kenya Washed Nyeri Gichathaini Factory AB",
                origin: "South-Central Kenya",
            },
            {
                name: "Haraaz Red",
                origin: "Yemen",
            },
            {
                name: "Ethiopia Natural Guji",
                origin: "Oromia Region",
            },
            {
                name: "Ethiopia Sidama Naia Bomb Natural Vertical Reserve",
                origin: "Oromia State",
            },
            {
                name: "Birambo Village DR Congo",
                origin: "Democratic Republic Of The Congo",
            },
        ]
    },
    {
        name: "WatchHouse Coffee",
        address: "125 Fenchurch Street, City",
        lat: 51.512,
        lng: -0.081,
        coffees: [
            {
                name: "Mocha Java Blend Fair Trade Organic",
                origin: "Sumatra",
            },
            {
                name: "Gesha Village Lot #85 Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Panama Boquete Torre Lot #1 Geisha Honey",
                origin: "Panama",
            },
            {
                name: "Ethiopia Dame Dabaye",
                origin: "Oromia Region",
            },
            {
                name: "Harimau Tiger Sumatra",
                origin: "Sumatra",
            },
            {
                name: "Ethiopian Sweet Lily",
                origin: "Oromia Region",
            },
            {
                name: "Natural Mexico Ixhuatlan del Cafe",
                origin: "Mexico",
            },
        ]
    },
    {
        name: "Origin Coffee Roasters",
        address: "65 Shoreditch High Street, Shoreditch",
        lat: 51.523,
        lng: -0.081,
        coffees: [
            {
                name: "Mirador Colombia",
                origin: "Southern Colombia",
            },
            {
                name: "Burundi Nemba",
                origin: "Kabuye",
            },
            {
                name: "Rum Barrel Aged Kauai Coffee",
                origin: "Kauai",
            },
            {
                name: "Colombia Antioqiua",
                origin: "Colombia",
            },
            {
                name: "Fruity Espresso Blend",
                origin: "Ethiopia",
            },
            {
                name: "Ethiopia Sidamo Natural Water Decaf",
                origin: "Oromia Region",
            },
            {
                name: "Ecuador Loja Hacienda La Papaya Bourbon Sidra 168hr Anaerobic Natural",
                origin: "Saraguro",
            },
            {
                name: "Kenya Nyeri Nudurutu Factory AB",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Duromina",
                origin: "Jimma Zone",
            },
        ]
    },
    {
        name: "Origin Coffee Roasters",
        address: "40-42 Great Titchfield Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Costa Rica Tarrazu Dota El Diosa Geisha",
                origin: "Costa Rica",
            },
            {
                name: "Kenya Kirinyaga Peaberry",
                origin: "South-Central Kenya",
            },
            {
                name: "Guatemala Single-Origin Espresso",
                origin: "Guatemala",
            },
            {
                name: "Kenya Thirikwa Single-Origin Espresso",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Sidama Bensa Farmer Tamiru 74158 Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Panama Finca Lerida Geisha Black Honey",
                origin: "Boquete Growing Region",
            },
            {
                name: "Finca San Luis Colombia Espresso",
                origin: "Center-West Colombia",
            },
            {
                name: "Trilogy",
                origin: "Africa",
            },
            {
                name: "Kenya Nyeri Muchoki Estate AA",
                origin: "South-Central Kenya",
            },
            {
                name: "Tres Dragones Colombia",
                origin: "Valle De Cauca Department",
            },
        ]
    },
    {
        name: "Climpson & Sons",
        address: "67 Broadway Market, Hackney",
        lat: 51.54,
        lng: -0.06,
        coffees: [
            {
                name: "Ethiopia Hambela Goro",
                origin: "Oromia Region",
            },
            {
                name: "Cream Tabby Espresso Blend",
                origin: "Kenya",
            },
            {
                name: "Kenya AA Top Lot",
                origin: "South-Central Kenya",
            },
            {
                name: "Sumatra Pantan Musara",
                origin: "Aceh Province",
            },
            {
                name: "Colombia Laboyano",
                origin: "Huila Department",
            },
            {
                name: "Panama Esmeralda Geisha Portón Oro Yeast",
                origin: "Panama",
            },
            {
                name: "Costa Rica Canet Raisin Honey",
                origin: "Costa Rica",
            },
            {
                name: "Ethiopia Reko",
                origin: "Southern Ethiopia",
            },
            {
                name: "Long Miles Coffee Project Nkonge Hill Burundi Red Honey",
                origin: "Burundi",
            },
        ]
    },
    {
        name: "Climpson & Sons",
        address: "Arch 374, Helmsley Place, Hackney",
        lat: 51.54,
        lng: -0.06,
        coffees: [
            {
                name: "Guo Mei Zhu",
                origin: "Colombia",
            },
            {
                name: "Colombia Cumbarco Lot 397",
                origin: "Valle De Cauca Department",
            },
            {
                name: "Brazil Red Catuaí Double-Anaerobic",
                origin: "Brazil",
            },
            {
                name: "Ethiopia Dimtu",
                origin: "Guji Zone",
            },
            {
                name: "José Flores El Salvador Natural",
                origin: "Northwestern El Salvador",
            },
            {
                name: "D31 The New Life",
                origin: "Sumatra",
            },
            {
                name: "Papua New Guinea Timuza",
                origin: "Papua New Guinea",
            },
            {
                name: "5a Sur",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Nude Espresso",
        address: "26 Hanbury Street, Spitalfields",
        lat: 51.52,
        lng: -0.075,
        coffees: [
            {
                name: "Kanyonyi Coffee Blend",
                origin: "Southwest Uganda",
            },
            {
                name: "Ethiopia Natural Guji Blue-Donkey 2020",
                origin: "Oromia Region",
            },
            {
                name: "Summer Night Blend Espresso",
                origin: "Honduras",
            },
            {
                name: "Gaitania Colombia",
                origin: "West-Central Colombia",
            },
            {
                name: "Guatemala Reserve Las Moritas Yellow Pacamara",
                origin: "Guatemala",
            },
            {
                name: "Colombia Cerro Azul Enano",
                origin: "Valle Del Cauca Department",
            },
            {
                name: "Nicaragua La Benedicion Pacamara",
                origin: "Jalapa",
            },
        ]
    },
    {
        name: "Nude Espresso",
        address: "Soho Square, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Malaysia Sabah Honey",
                origin: "East Malaysia",
            },
            {
                name: "Ka’u IPA Natural",
                origin: "Big Island Of Hawai’I",
            },
            {
                name: "Finca Sophia Gesha Washed",
                origin: "Chiriquí",
            },
            {
                name: "Colombia Monteblanco Gesha Cold Fermentation",
                origin: "Huila Department",
            },
            {
                name: "Ardent Ethiopia Natural",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Colombia La Gallera Estate",
                origin: "Colombia",
            },
            {
                name: "Thailand Robusta Panid Choosit Espresso",
                origin: "Chumphon",
            },
            {
                name: "El Salvador Monte Verde",
                origin: "Santa Ana Department",
            },
        ]
    },
    {
        name: "The Roasting Party",
        address: "1-3 Dray Walk, Brick Lane",
        lat: 51.52,
        lng: -0.075,
        coffees: [
            {
                name: "Kona Pink Honey Black Rock Farm",
                origin: "Big Island Of Hawai’I",
            },
            {
                name: "Ethiopia Shakiso Mormora",
                origin: "Southern Ethiopia",
            },
            {
                name: "Kenya Kapsakiso",
                origin: "Western Kenya",
            },
            {
                name: "Camilina Geisha",
                origin: "Chiriqui Province",
            },
            {
                name: "Colombia Antioquia Magico",
                origin: "Antioquia Department",
            },
            {
                name: "Yemen Matari",
                origin: "Yemen",
            },
            {
                name: "Karamo Ethiopia",
                origin: "Ethiopia",
            },
            {
                name: "Lone Peak Cafe Series Blend",
                origin: "Costa Rica",
            },
            {
                name: "Benti Nenqa #209 Ethiopia",
                origin: "Oromia Region",
            },
            {
                name: "Esmeralda Estate Panama Geisha",
                origin: "Western Panama",
            },
        ]
    },
    {
        name: "The Roasting Party",
        address: "146 Brick Lane, Shoreditch",
        lat: 51.52,
        lng: -0.075,
        coffees: [
            {
                name: "Nicaragua Selva Negra",
                origin: "Nicaragua",
            },
            {
                name: "Maui Kupa’a Orange Bourbon",
                origin: "Island Of Maui",
            },
            {
                name: "Ethiopia Konga Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia COE 2nd place Washed Sidamo Rumudamo",
                origin: "Sidama (Also Sidamo) Growing Region",
            },
            {
                name: "Angamaza",
                origin: "Loja Province",
            },
            {
                name: "COE Brazil Naturals 2017 Sitio Esperanza 7th place",
                origin: "Southern Minas Gerais State",
            },
        ]
    },
    {
        name: "Coffee Island",
        address: "45-47 Old Street, Shoreditch",
        lat: 51.523,
        lng: -0.081,
        coffees: [
            {
                name: "Snowfield Espresso Blend",
                origin: "Ethiopia",
            },
            {
                name: "Colombia La Siria Geisha",
                origin: "Huila Department",
            },
            {
                name: "Ethiopia Hambela Hasam",
                origin: "Guji Zone",
            },
            {
                name: "Ethiopia Gora Kone Sidamo",
                origin: "South-Central Ethiopia",
            },
            {
                name: "El Salvador Pacamara",
                origin: "El Salvador",
            },
            {
                name: "Kenya Embu Uteuzi Jimbo",
                origin: "Eastern Province Kenya",
            },
            {
                name: "Kenya Kirinyaga",
                origin: "Tarrazu",
            },
        ]
    },
    {
        name: "Coffee Island",
        address: "123 Old Street, Shoreditch",
        lat: 51.523,
        lng: -0.081,
        coffees: [
            {
                name: "Ethiopia Durato Bombe Natural",
                origin: "Sidama Region",
            },
            {
                name: "Static Peru Valle de Chingama Peru",
                origin: "Northern Peru",
            },
            {
                name: "Honey Typica",
                origin: "Big Island Of Hawai'I",
            },
            {
                name: "Kona SL34 Champagne Natural Uluwehi Farm",
                origin: "Kona",
            },
            {
                name: "Espresso Classico",
                origin: "Ecuador",
            },
            {
                name: "Pantan Musara Sumatra",
                origin: "Sumatra",
            },
        ]
    },
    {
        name: "Brew Lab",
        address: "6-8 South College Street, Edinburgh",
        lat: 55.948,
        lng: -3.187,
        coffees: [
            {
                name: "Congo Muungano",
                origin: "Democratic Republic Of The Congo",
            },
            {
                name: "Colombia La Victoria",
                origin: "Colombia",
            },
            {
                name: "Pena Blanca Black Honey",
                origin: "Comayagua",
            },
            {
                name: "Addis Katema",
                origin: "Gedeo Zone",
            },
            {
                name: "Kabiufa Papua New Guinea",
                origin: "Eastern Highlands",
            },
            {
                name: "Ethiopia Hambela Yellow Honey James Selection",
                origin: "Oromia Region",
            },
        ]
    },
    {
        name: "Artisan Coffee",
        address: "123 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "El Salvador Finca Santa Elena El Corzo Pacamara",
                origin: "El Salvador",
            },
            {
                name: "Costa Rica Yellow Honey Kinka Bear Lot 20-02",
                origin: "Guji Zone",
            },
            {
                name: "Costa Rica Brunca Rivense La Guaca Passion Honey",
                origin: "Costa Rica",
            },
            {
                name: "Amasia Blend",
                origin: "Guatemala",
            },
            {
                name: "Taiwan Natural Alishan Ching-Ye Farm",
                origin: "Chia-Yi",
            },
            {
                name: "Ethiopia Gede Natural",
                origin: "Guji Zone",
            },
            {
                name: "Philippines Sitio Belis 1017",
                origin: "Philippines",
            },
            {
                name: "Kona Pumpkin Spice Anaerobic",
                origin: "Big Island Of Hawai’I",
            },
            {
                name: "Ethiopia Guji Hambela Wate 74110/74112/74158 Mini Natural",
                origin: "Oromia Region",
            },
            {
                name: "Espresso Blend",
                origin: "Ethiopia",
            },
        ]
    },
    {
        name: "Brew & Bake",
        address: "45 Camden High Street, Camden",
        lat: 51.54,
        lng: -0.14,
        coffees: [
            {
                name: "Rock the House Blend",
                origin: "South America",
            },
            {
                name: "Colombia Buesaco",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Yirgacheffe G1 Idido Natural",
                origin: "Snnp Region",
            },
            {
                name: "Kenya Rungeto",
                origin: "South-Central Kenya",
            },
            {
                name: "Maypop",
                origin: "Colombia",
            },
            {
                name: "Colombia Laderas del Tapias Estate Natural Geisha",
                origin: "Caldas Department",
            },
            {
                name: "Guatemala Hunapu",
                origin: "Guatemala",
            },
            {
                name: "Bright House Signature Coffee",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Café Nero",
        address: "78 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Sumatra Lintong",
                origin: "North Sumatra Province",
            },
            {
                name: "Gahahe Burundi Natural",
                origin: "Kabuye",
            },
            {
                name: "Panama Hacienda La Esmeralda Super Mario Geisha",
                origin: "Western Panama",
            },
            {
                name: "El Salvador Pacamara Honey",
                origin: "El Salvador",
            },
            {
                name: "Colombia Pink Bourbon El Corazon",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Deri Kochoha",
                origin: "Oromia Region",
            },
        ]
    },
    {
        name: "Café Nero",
        address: "156 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Static Kenya Kavutiri",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Guji Mormora Organic Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Costa Rica Brunca Rivense La Guaca Passion Honey",
                origin: "Costa Rica",
            },
            {
                name: "Kenya AA Kirinyaga",
                origin: "Sidama (Also Sidamo) Growing Region",
            },
            {
                name: "Ethiopia Birbisakela Washed G1",
                origin: "Southern Ethiopia",
            },
            {
                name: "Peru Yanesha Geisha Washed Anaerobic",
                origin: "Villa Rica",
            },
            {
                name: "Alemu Bukato Ethiopia",
                origin: "Southern Ethiopia",
            },
            {
                name: "Indonesian Island Blend",
                origin: "Flores And Sumatra",
            },
            {
                name: "Classic MK Blend",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Café Nero",
        address: "89 Tottenham Court Road, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Honduras Reserve Luis Nolasco Nanolot",
                origin: "Honduras",
            },
            {
                name: "Ethiopia Guji Hambela Wate Natural G1",
                origin: "Oromia Region",
            },
            {
                name: "Hawai’i Kilauea Volcano Yeast Fermentation Washed",
                origin: "Big Island Of Hawai’I",
            },
            {
                name: "Kenya Kiambu AB Uklili 1905 Lot",
                origin: "South-Central Kenya",
            },
            {
                name: "Tres Dragones Colombia",
                origin: "Valle De Cauca Department",
            },
            {
                name: "Ecuador Quilanga Typica Yeast Fermentation",
                origin: "Ecuador",
            },
            {
                name: "Ethiopia Cup of Excellence Guji Natural",
                origin: "Hambela Wamena Woreda",
            },
            {
                name: "Colombia Castillo Fruit Maceration Series (Strawberry)",
                origin: "Quindio Department",
            },
            {
                name: "Kenya Nyeri Othaya Gura AA",
                origin: "South-Central Kenya",
            },
        ]
    },
    {
        name: "Costa Coffee",
        address: "125 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Zambia Kateshi Natural Process",
                origin: "Sidamo (Also Sidama) Growing Region",
            },
            {
                name: "Zambia AAA/AA",
                origin: "Ethiopia",
            },
            {
                name: "Kenya Nyeri",
                origin: "South-Central Kenya",
            },
            {
                name: "Crystalina",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Kochere Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Ethiopia Banko Gotiti",
                origin: "Yirgacheffe Growing Region",
            },
            {
                name: "Nicaragua Women Producers",
                origin: "Jinotega",
            },
            {
                name: "Hawaiian Kona 100%",
                origin: "Big Island Of Hawai'I",
            },
            {
                name: "100% Kona Bourbon Pointu Laurina",
                origin: "North Kona Growing District",
            },
        ]
    },
    {
        name: "Costa Coffee",
        address: "67 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Crescent Blend",
                origin: "Nicaragua",
            },
            {
                name: "Geisha",
                origin: "“Big Island” Of Hawai’I",
            },
            {
                name: "Honduras COMSA Natural",
                origin: "La Paz Department",
            },
            {
                name: "Ethiopia Guji Uraga Tebe Burka Natural G1",
                origin: "Oromia Region",
            },
            {
                name: "Umoja Organic Red Bourbon",
                origin: "Democratic Republic Of The Congo",
            },
            {
                name: "Sumatra Dark",
                origin: "Indonesia",
            },
            {
                name: "Panama Finca Deborah Afterglow Geisha Natural",
                origin: "Chiriquí Province",
            },
            {
                name: "Santa Elena Colombia Caturra Natural",
                origin: "Colombia",
            },
        ]
    },
    {
        name: "Costa Coffee",
        address: "234 Tottenham Court Road, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Magnolia Blend Espresso",
                origin: "Costa Rica",
            },
            {
                name: "Kenya Ndnunduri",
                origin: "South-Central Kenya",
            },
            {
                name: "Kona Maragogype Pink Honey",
                origin: "Kona District",
            },
            {
                name: "Static Kenya Kavutiri",
                origin: "South-Central Kenya",
            },
            {
                name: "Dharma Espresso Blend",
                origin: "Ethiopia",
            },
        ]
    },
    {
        name: "Starbucks",
        address: "156 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Brazil Legender Sitio Taquara Natural",
                origin: "Minas Gerais State",
            },
            {
                name: "Ethiopia Wegida Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Umoja Organic Red Bourbon",
                origin: "Democratic Republic Of The Congo",
            },
            {
                name: "Honduras Los Catadores Honey",
                origin: "La Paz Department",
            },
            {
                name: "Three Queens Blend",
                origin: "Burundi",
            },
            {
                name: "Sudan Rume",
                origin: "Colombia",
            },
            {
                name: "Guji Ethiopia Washed",
                origin: "Oromia Region",
            },
        ]
    },
    {
        name: "Starbucks",
        address: "89 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Panama Ninety Plus Perci Lot 50",
                origin: "Far Western Panama",
            },
            {
                name: "Empress Kenya",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Guji Buku",
                origin: "Guji Zone",
            },
            {
                name: "Ethiopia Washed Shantawene",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Congo Boza",
                origin: "Drc Congo",
            },
            {
                name: "Ethiopia Guji Sakicha JH144",
                origin: "Oromia Region",
            },
            {
                name: "Brazil Ipanema Golden Edition C26 Lychee",
                origin: "Brazil",
            },
        ]
    },
    {
        name: "Starbucks",
        address: "345 Tottenham Court Road, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Tigesit Waqa",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Ethiopia Guji Hambela Dabaye",
                origin: "Oromia Region",
            },
            {
                name: "Jade Espresso",
                origin: "Yirgacheffe Growing Region",
            },
            {
                name: "Sweet Collection Blend",
                origin: "Guatemala",
            },
            {
                name: "Brazil Ipanema Golden Edition C26 Lychee",
                origin: "Brazil",
            },
            {
                name: "Hero Blend Espresso",
                origin: "Ethiopia",
            },
            {
                name: "Bright Minds",
                origin: "Caldas",
            },
        ]
    },
    {
        name: "Pret A Manger",
        address: "234 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Kona SL34 Champagne Natural Uluwehi Farm",
                origin: "Kona",
            },
            {
                name: "Dodora Double",
                origin: "Yirgacheffe Growing Region",
            },
            {
                name: "Sweet Holiday Blend",
                origin: "Guatemala",
            },
            {
                name: "Ethiopia Yirgacheffe Adorsi G1",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Kenya Gura AA",
                origin: "Lintong Growing Region",
            },
            {
                name: "Sustainable Harvest Homacho Waeno Natural",
                origin: "Sidama (Also Sidamo) Growing Region",
            },
            {
                name: "Rwanda Simbi",
                origin: "Rwanda",
            },
            {
                name: "Coffea Diversa Rume Sudan Winey",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Pret A Manger",
        address: "123 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "El Salvador El Matazano Pacamara",
                origin: "El Salvador",
            },
            {
                name: "Burundi Washed “Crystal”",
                origin: "Katanga",
            },
            {
                name: "BNT Espresso Blend",
                origin: "Guatemala",
            },
            {
                name: "Kenya Washed Nyeri Gichathaini Factory AB",
                origin: "South-Central Kenya",
            },
            {
                name: "Ethiopia Nano Genji",
                origin: "Jimma Zone",
            },
            {
                name: "Pandemic Pacamara Blend",
                origin: "Guatemala",
            },
            {
                name: "Madagascar Yellow Bourbon Santatra Coop",
                origin: "Odo Shakiso District",
            },
            {
                name: "La Union (Colombia)",
                origin: "Southern Colombia",
            },
            {
                name: "Tablon La Cima",
                origin: "El Salvador",
            },
            {
                name: "Medium-Dark Roast",
                origin: "North Kona",
            },
        ]
    },
    {
        name: "Pret A Manger",
        address: "456 Tottenham Court Road, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Lover’s Quarrel",
                origin: "Indonesia",
            },
            {
                name: "Guatemala Buena Vista Single-Origin Espresso",
                origin: "Guatemala",
            },
            {
                name: "Panama Perci Geisha Natural",
                origin: "Panama",
            },
            {
                name: "Strawberry Forest",
                origin: "Nicaragua",
            },
            {
                name: "Blossom Single Origin Espresso",
                origin: "Southwestern Ethiopia",
            },
            {
                name: "Kenya Nyeri Maganjo AB",
                origin: "South-Central Kenya",
            },
        ]
    },
    {
        name: "Gail's Bakery",
        address: "67 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Ethiopia Dry-Process Guji Hambela",
                origin: "Oromia Region",
            },
            {
                name: "Costa Rica Esteban Zamora",
                origin: "Costa Rica",
            },
            {
                name: "Lycello by Ninety Plus",
                origin: "Far Western Panama",
            },
            {
                name: "Costa Rica Cordillera de Fuego",
                origin: "Costa Rica",
            },
            {
                name: "Ka’u Yellow Bourbon",
                origin: "Big Island Of Hawai'I",
            },
            {
                name: "Ethiopia Bedhatu Washed",
                origin: "Gedeo Zone",
            },
            {
                name: "Guatemala CODECH Women’s Lot",
                origin: "Huehuetenango Department",
            },
            {
                name: "Guatemala Finca Las Terrazas",
                origin: "Guatemala",
            },
            {
                name: "Ecuador Pichincha Typica",
                origin: "Northern Ecuador",
            },
        ]
    },
    {
        name: "Gail's Bakery",
        address: "234 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Guatemala Finca Granada Natural",
                origin: "Huehuetenango Department",
            },
            {
                name: "Ethiopia Suke Quto",
                origin: "Oromia Region",
            },
            {
                name: "Whiskey Dreams Moka Java",
                origin: "Ethiopia",
            },
            {
                name: "Kenya Athena AB Espresso",
                origin: "South-Central Kenya",
            },
            {
                name: "Pandemic Pacamara Blend",
                origin: "Guatemala",
            },
            {
                name: "Ethiopia Banko Gotiti",
                origin: "Yirgacheffe Growing Region",
            },
            {
                name: "Burundi Buhorwa",
                origin: "Burundi",
            },
        ]
    },
    {
        name: "Paul",
        address: "345 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Kenya Chorongi Peaberry",
                origin: "Nyeri Growing Region",
            },
            {
                name: "Static Peru Valle de Chingama Peru",
                origin: "Northern Peru",
            },
            {
                name: "Malaysia Sabah Honey",
                origin: "East Malaysia",
            },
            {
                name: "Ethiopia Yirgacheffe Cleopatra Natural",
                origin: "Southern Ethiopia",
            },
            {
                name: "Guo Mei Zhu",
                origin: "Colombia",
            },
        ]
    },
    {
        name: "Paul",
        address: "456 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Alemu Bukato Ethiopia",
                origin: "Southern Ethiopia",
            },
            {
                name: "Guatemala Washed El General Lot SummerEnd 2020",
                origin: "Central Guatemala",
            },
            {
                name: "Rock the House Blend",
                origin: "South America",
            },
            {
                name: "Organic 18 Rabbits Honduras",
                origin: "Honduras",
            },
            {
                name: "SL28",
                origin: "“Big Island” Of Hawai’I",
            },
            {
                name: "Colombia Antioqiua",
                origin: "Colombia",
            },
            {
                name: "Kenya Murang’a Githiga AA",
                origin: "Central Province",
            },
            {
                name: "Mahonda Burundi",
                origin: "Burundi",
            },
            {
                name: "Taiwan Natural Alishan TFU’YA Kakalove Lot",
                origin: "Chia-Yi",
            },
            {
                name: "Aged Sumatra Semiga",
                origin: "Sumatra",
            },
        ]
    },
    {
        name: "Eat",
        address: "567 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Colombia Cerro Azul Enano",
                origin: "Valle Del Cauca Department",
            },
            {
                name: "Ethiopia",
                origin: "Southern Ethiopia",
            },
            {
                name: "Costa Rica Cattleya Anaerobic",
                origin: "Central Valley",
            },
            {
                name: "Ethiopia Yirgacheffe Kochere",
                origin: "Southern Ethiopia",
            },
            {
                name: "Gahahe Burundi Natural",
                origin: "Kabuye",
            },
            {
                name: "Kimi Blend",
                origin: "Colombia",
            },
            {
                name: "Monarch Estate Hawaiian Gesha",
                origin: "North Kona Growing District",
            },
            {
                name: "Eaagads Estate Kenya",
                origin: "South-Central Kenya",
            },
        ]
    },
    {
        name: "Eat",
        address: "678 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Kenya Kirinyaga Mukangu AB",
                origin: "South-Central Kenya",
            },
            {
                name: "Dominican Republic Organic Ramirez Estate",
                origin: "Dominican Republic",
            },
            {
                name: "Ethiopia “Ear Candy” Uraga Guji",
                origin: "Oromia Region",
            },
            {
                name: "Galapagos La Tortuga",
                origin: "Galapagos Islands",
            },
            {
                name: "Ethiopia Aphrodite Washed Espresso",
                origin: "South-Central Ethiopia",
            },
        ]
    },
    {
        name: "Itsu",
        address: "789 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Guatemala Acatenango",
                origin: "Guatemala",
            },
            {
                name: "Ethiopia Karamo Natural",
                origin: "South-Central Ethiopia",
            },
            {
                name: "Guatemala Don Angel",
                origin: "Huehuetenango Department",
            },
            {
                name: "Snowfield Espresso Blend",
                origin: "Ethiopia",
            },
            {
                name: "Sidamo Suke Quto",
                origin: "Southern Ethiopia",
            },
            {
                name: "Aces La Juntas",
                origin: "San Agustín",
            },
            {
                name: "Guatemala Acatenango Gesha",
                origin: "Guatemala",
            },
            {
                name: "Titan Blend (FTO)",
                origin: "Colombia",
            },
        ]
    },
    {
        name: "Itsu",
        address: "890 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Panama Auromar Camilina Geisha Natural",
                origin: "Chiriqui Province",
            },
            {
                name: "Philippines Sitio Belis",
                origin: "Philippines",
            },
            {
                name: "Maui Kupa’a Orange Bourbon",
                origin: "Island Of Maui",
            },
            {
                name: "GA 400 Espresso",
                origin: "Colombia",
            },
            {
                name: "Ethiopia Botabaa",
                origin: "Gedeo Zone",
            },
            {
                name: "Kenya Gichithaini AA",
                origin: "Nyeri County",
            },
            {
                name: "Kenya Kariru PB",
                origin: "South-Central Kenya",
            },
            {
                name: "DR Congo Muungano Co-op South Kivu",
                origin: "Democratic Republic Of The Congo",
            },
            {
                name: "Panama Esmeralda Geisha Portón Oro Yeast",
                origin: "Panama",
            },
            {
                name: "Guatemala Guaya’B Coop",
                origin: "Guatemala",
            },
        ]
    },
    {
        name: "Wasabi",
        address: "901 Oxford Street, Fitzrovia",
        lat: 51.52,
        lng: -0.14,
        coffees: [
            {
                name: "Honey Typica Anaerobic",
                origin: "“Big Island” Of Hawai’I",
            },
            {
                name: "Kona SL34 Champagne Natural Uluwehi Farm",
                origin: "Kona",
            },
            {
                name: "Kasasagi",
                origin: "Costa Rica",
            },
            {
                name: "Peru La Salina",
                origin: "Cajamarca Region",
            },
            {
                name: "Colombia Sidra",
                origin: "Colombia",
            },
            {
                name: "Gaitania Colombia",
                origin: "West-Central Colombia",
            },
            {
                name: "Ethiopia Buku Abela",
                origin: "Oromia Region",
            },
            {
                name: "Colombia Tolima",
                origin: "Colombia",
            },
        ]
    },
    {
        name: "Wasabi",
        address: "012 Regent Street, Soho",
        lat: 51.514,
        lng: -0.136,
        coffees: [
            {
                name: "Taiwan Alluring Scent Estate Natural",
                origin: "Taiwan",
            },
            {
                name: "Honduras La Lesquinada Washed",
                origin: "Honduras",
            },
            {
                name: "Flora Blend Espresso",
                origin: "Asia Pacific",
            },
            {
                name: "Foundry Blend",
                origin: "Burundi",
            },
            {
                name: "Taiwan BA LU NA",
                origin: "Taiwan",
            },
        ]
    },
];