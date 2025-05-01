import pickle

with open("skin_tone_model.pkl","rb")as f:
    model = pickle.load(f)

def recommend_colors(skin_tone):
    color_map ={
        "Fair": [
            {"name":"Pastel pink" ,"hex": "#FFD1DC"},
            {"name":"Sky blue","hex":"#87CEEB"},
            {"name":"Lavender","hex":"#E6E6FA"},
            {"name":"Soft yellow","hex":"#FFFFE0"}
                ],
        "Light":[
            {"name":"Peach","hex":"#FFE5B4"},
            {"name":"Coral","hex":"#FF7F50"},
            {"name":"Turquoise","hex":"#40E0D0"},
            {"name":"Mint green","hex":"#98FF98"}
                ],
        "Medium":[
            {"name":"Emerald","hex":"#50C878"},
            {"name":"Ruby","hex":"#E0115F"},
            {"name":"Burnt orange","hex":"#CC5500"},
            {"name":"Mustard","hex":"#FFDB58"}
                ],
        "Olive":[
            {"name":"Navy","hex":"#000080"},
            {"name":"Forest green","hex":"#228B22"},
            {"name":"Maroon","hex":"#800000"},
            {"name":"Bronze","hex":"#CD7F32"}
                ],
        "Dark": [
            {"name":"White","hex":"#FFFFFF"},
            {"name":"Gold","hex":"#FFD700"},
            {"name":"Vibrant yellow","hex":"#FFEA00"},
            {"name":"Cobalt blue","hex":"#0047AB"}
                ]
    }
    return color_map.get(skin_tone,[
            {"name":"Black","hex":"#000000"},
            {"name":"White","hex":"#FFFFFF"}
        ])

def predict_skin_tone(r,g,b):
    tone = model.predict([[r,g,b]])[0]
    tone = tone.capitalize()
    colors = recommend_colors(tone)
    return tone,colors

