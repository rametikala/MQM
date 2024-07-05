import json
data = []
gg = ['O','Mistranslation-Minor', 'Addition-Critical', 'Omission-Critical', 'Spelling-Neutral', 'Part_of_speech-Major', 'Omission-Major', 'Mistranslation-Critical', 'TAM-Minor', 'Mistranslation-Neutral', 'Addition-Neutral', 'Anaphora/coreference/lexical_cohesion-Neutral', 'GNP-Minor', 'Part_of_speech-Minor', 'Addition-Major', 'TAM-Neutral', 'Function_words-Major', 'Anaphora/coreference/lexical_cohesion-Minor', 'Spelling-Minor', 'Unidiometic_style-Minor', 'Anaphora/coreference/lexical_cohesion-Major', 'GNP-Critical', 'Part_of_speech-Critical', 'Spelling-Major', 'Omission-Minor', 'Terminology-Major', 'Omission-Neutral', 'Orthography', 'TAM-Major', 'Addition-Minor', 'term="', 'TAM-Critical', 'Mistranslation-Major', 'GNP-Major', 'Coherence-Major', 'Anaphora/coreference/lexical_cohesion-Critical', 'Unidiometic_style-Major', 'Spelling-Critical', 'Connectives-Major']


def parse_file(file_path):
    texts = []
    bn_texts = []
    bn_error_labels = []
    bn_text = ''

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            # filess.writelines('"'+f"{parts[0]}"+'"')
            # en_texts.append(parts[0].strip())
            bn_parts = parts[-1].strip()
            bn_partss = bn_parts.strip().split(' ')
            en_texts = parts[0].strip()
            en_partss = en_texts.strip().split(' ')
            for new_en in en_partss:
                bn_error_labels.append(0)
            
            # for new_parts in bn_partss:
            #     bn,rn = new_parts.strip().split('|||')
            #     bn_text = bn_text +' '+bn
            #     for i in range(len(gg)):
            #         if(rn == gg[i]):
            #             bn_error_labels.append(i)

            for new_parts in bn_partss:
                bn,rn = new_parts.strip().split('|||')
                bn_text = bn_text +' '+bn
                if(len(rn)>3):
                    bn_error_labels.append(rn)
                else:
                    bn_error_labels.append(0)

            
            final_text = parts[0].strip() +' '+bn_text.strip()
            # texts.append(final_text)
            # bn_texts.append(bn_text)



            entry = {
                "text": final_text,
                "error_labels": bn_error_labels  # Assuming 12 error labels as in the example
            }

            bn_error_labels = []
            final_text = ''
            bn_text = ''


            data.append(entry)




parse_file('MQM_data_eng_ban.txt')
with open("data2.json", 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=2)

print(f"Data has been successfully converted and saved ")


    