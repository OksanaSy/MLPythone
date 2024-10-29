import nltk
from nltk.corpus import conll2002
import spacy
from spacy.training import Example
from sklearn.metrics import accuracy_score

nltk.download('conll2002')

nlp_spacy = spacy.blank("es") 

nltk_train_data = conll2002.iob_sents('esp.train')
nltk_test_data = conll2002.iob_sents('esp.testb')

def prepare_training_data(nltk_data):
    examples = []
    for sent in nltk_data:
        tokens = [token for token, _, _ in sent] 
        tags = [tag for _, tag, _ in sent]

        print(f"Tokens: {tokens}")  
        print(f"Tags: {tags}") 
        print(f"Token count: {len(tokens)}, Tag count: {len(tags)}")  

        if len(tokens) == len(tags): 
            doc = nlp_spacy.make_doc(" ".join(tokens))
            examples.append(Example.from_dict(doc, {"tags": tags}))
        else:
            print(f"Skipping sentence due to length mismatch: {len(tokens)} tokens and {len(tags)} tags")
    return examples

train_examples = prepare_training_data(nltk_train_data)

tagger = nlp_spacy.add_pipe("tagger") 

tagger.add_label("B-PER")
tagger.add_label("I-PER")
tagger.add_label("B-LOC")
tagger.add_label("I-LOC")
tagger.add_label("B-ORG")
tagger.add_label("I-ORG")
tagger.add_label("B-MISC")
tagger.add_label("I-MISC")

optimizer = nlp_spacy.begin_training()
for i in range(10): 
    for example in train_examples:
        nlp_spacy.update([example], drop=0.5, losses={})

def evaluate_tagger(nlp, test_data, lib_name="spaCy"):
    true_tags = []
    pred_tags = []

    for sent in test_data:
        tokens = [token for token, _, _ in sent]
        true_tags.extend([tag for _, tag, _ in sent])

        doc = nlp(" ".join(tokens))
        pred_tags.extend([token.tag_ for token in doc]) 

    accuracy = accuracy_score(true_tags, pred_tags)
    print(f"Accuracy of {lib_name} tagger: {accuracy:.4f}")

evaluate_tagger(nlp_spacy, nltk_test_data, lib_name="spaCy")


"""
Tokens: ['Melbourne', '(', 'Australia', ')', ',', '25', 'may', '(', 'EFE', ')', '.']
Tags: ['NP', 'Fpa', 'NP', 'Fpt', 'Fc', 'Z', 'NC', 'Fpa', 'NC', 'Fpt', 'Fp']
Token count: 11, Tag count: 11
Tokens: ['-']
Tags: ['Fg']
Token count: 1, Tag count: 1
Tokens: ['El', 'Abogado', 'General', 'del', 'Estado', ',', 'Daryl', 'Williams', ',', 'subrayó', 'hoy', 'la', 'necesidad', 'de', 'tomar', 'medidas', 'para', 'proteger', 'al', 'sistema', 'judicial', 'australiano', 'frente', 'a', 'una', 'página', 'de', 'internet', 'que', 'imposibilita', 'el', 'cumplimiento', 'de', 'los', 'principios', 'básicos', 'de', 'la', 'Ley', '.']
Tags: ['DA', 'NC', 'AQ', 'SP', 'NC', 'Fc', 'VMI', 'NC', 'Fc', 'VMI', 'RG', 'DA', 'NC', 'SP', 'VMN', 'NC', 'SP', 'VMN', 'SP', 'NC', 'AQ', 'AQ', 'RG', 'SP', 'DI', 'NC', 'SP', 'NC', 'PR', 'VMI', 'DA', 'NC', 'SP', 'DA', 'NC', 'AQ', 'SP', 'DA', 'NC', 'Fp']
Token count: 40, Tag count: 40
Tokens: ['La', 'petición', 'del', 'Abogado', 'General', 'tiene', 'lugar', 'después', 'de', 'que', 'un', 'juez', 'del', 'Tribunal', 'Supremo', 'del', 'estado', 'de', 'Victoria', '(', 'Australia', ')', 'se', 'viera', 'forzado', 'a', 'disolver', 'un', 'jurado', 'popular', 'y', 'suspender', 'el', 'proceso', 'ante', 'el', 'argumento', 'de', 'la', 'defensa', 'de', 'que', 'las', 'personas', 'que', 'lo', 'componían', 'podían', 'haber', 'obtenido', 'información', 'sobre', 'el', 'acusado', 'a', 'través', 'de', 'la', 'página', 'CrimeNet', '.']
Tags: ['DA', 'NC', 'SP', 'NC', 'AQ', 'VMI', 'NC', 'RG', 'SP', 'CS', 'DI', 'NC', 'SP', 'NC', 'AQ', 'SP', 'NC', 'SP', 'NC', 'Fpa', 'NP', 'Fpt', 'P0', 'VMS', 'AQ', 'SP', 'VMN', 'DI', 'NC', 'AQ', 'CC', 'VMN', 'DA', 'NC', 'SP', 'DA', 'NC', 'SP', 'DA', 'NC', 'SP', 'CS', 'DA', 'NC', 'PR', 'PP', 'VMI', 'VMI', 'VAN', 'VMP', 'NC', 'SP', 'DA', 'VMP', 'SP', 'NC', 'SP', 'DA', 'NC', 'AQ', 'Fp']
Token count: 61, Tag count: 61
Tokens: ['Esta', 'página', 'web', 'lleva', 'un', 'mes', 'de', 'existencia', ',', 'tiempo', 'en', 'el', 'que', 'ha', 'sido', 'visitada', 'en', 'más', 'de', 'un', 'millón', 'de', 'ocasiones', ',', 'y', 'facilita', 'información', 'sobre', 'miles', 'de', 'crímenes', 'y', 'criminales', 'ya', 'enjuiciados', 'o', 'aún', 'perseguidos', ',', 'datos', 'que', 'salen', 'de', 'artículos', 'de', 'periódicos', 'y', 'archivos', 'judiciales', '.']
Tags: ['DD', 'NC', 'AQ', 'VMI', 'DI', 'NC', 'SP', 'NC', 'Fc', 'NC', 'SP', 'DA', 'PR', 'VAI', 'VSP', 'VMP', 'SP', 'RG', 'SP', 'DI', 'NC', 'SP', 'NC', 'Fc', 'CC', 'VMI', 'NC', 'SP', 'PN', 'SP', 'NC', 'CC', 'VMM', 'RG', 'VMP', 'CC', 'RG', 'VMM', 'Fc', 'NC', 'PR', 'VMI', 'SP', 'NC', 'SP', 'NC', 'CC', 'NC', 'AQ', 'Fp']
Token count: 50, Tag count: 50
Tokens: ['Por', 'su', 'parte', ',', 'el', 'Abogado', 'General', 'de', 'Victoria', ',', 'Rob', 'Hulls', ',', 'indicó', 'que', 'no', 'hay', 'nadie', 'que', 'controle', 'que', 'las', 'informaciones', 'contenidas', 'en', 'CrimeNet', 'son', 'veraces', '.']
Tags: ['SP', 'DP', 'NC', 'Fc', 'DA', 'NC', 'AQ', 'SP', 'NC', 'Fc', 'NC', 'AQ', 'Fc', 'VMI', 'CS', 'RN', 'VAI', 'PI', 'PR', 'VMS', 'CS', 'DA', 'NC', 'AQ', 'SP', 'NC', 'VSI', 'AQ', 'Fp']
Token count: 29, Tag count: 29
Tokens: ['Hulls', 'señaló', 'que', 'en', 'el', 'sistema', 'jurídico', 'de', 'la', 'Commonwealth', ',', 'en', 'el', 'que', 'se', 'basa', 'la', 'justicia', 'australiana', ',', 'es', 'fundamental', 'que', 'la', 'persona', 'sea', 'juzgada', 'únicamente', 'teniendo', 'en', 'cuenta', 'las', 'pruebas', 'presentadas', 'ante', 'el', 'juez', '.']
Tags: ['NP', 'VMI', 'CS', 'SP', 'DA', 'NC', 'AQ', 'SP', 'DA', 'NC', 'Fc', 'SP', 'DA', 'PR', 'P0', 'VMI', 'DA', 'NC', 'AQ', 'Fc', 'VSI', 'AQ', 'PR', 'DA', 'NC', 'VSS', 'VMP', 'RG', 'VMG', 'SP', 'NC', 'DA', 'NC', 'AQ', 'SP', 'DA', 'NC', 'Fp']
Token count: 38, Tag count: 38
Tokens: ['La', 'citada', 'página', 'de', 'internet', 'se', 'considera', 'también', 'la', 'causa', 'de', 'la', 'conclusión', 'prematura', 'en', 'un', 'caso', 'de', 'asesinato', 'juzgado', 'recientemente', 'en', 'la', 'ciudad', 'de', 'Melbourne', ',', 'capital', 'del', 'estado', 'de', 'Victoria', '.']
Tags: ['DA', 'AQ', 'NC', 'SP', 'NC', 'P0', 'VMI', 'RG', 'DA', 'NC', 'SP', 'DA', 'NC', 'AQ', 'SP', 'DI', 'NC', 'SP', 'NC', 'VMP', 'RG', 'SP', 'DA', 'NC', 'SP', 'NC', 'Fc', 'AQ', 'SP', 'NC', 'SP', 'NC', 'Fp']
Token count: 33, Tag count: 33
Tokens: ['Santander', ',', '25', 'may', '(', 'EFE', ')', '.']
Tags: ['NC', 'Fc', 'Z', 'NC', 'Fpa', 'NC', 'Fpt', 'Fp']
Token count: 8, Tag count: 8
Tokens: ['-']
Tags: ['Fg']
Token count: 1, Tag count: 1
Tokens: ['Izquierda', 'Unida', 'de', 'Santander', 'presentó', 'hoy', 'su', 'nuevo', 'boletín', 'trimestral', '"', 'Ciudad', '"', ',', 'una', 'publicación', 'de', 'la', 'que', 'se', 'distribuirán', '10.000', 'ejemplares', 'preferentemente', 'en', 'los', 'barrios', 'del', 'municipio', 'donde', 'la', 'colación', 'cuenta', 'con', 'mayor', 'respaldo', 'ciudadano', '.']
Tags: ['NC', 'AQ', 'SP', 'NC', 'VMI', 'RG', 'DP', 'AQ', 'NC', 'AQ', 'Fe', 'NC', 'Fe', 'Fc', 'DI', 'NC', 'SP', 'DA', 'PR', 'P0', 'VMI', 'Z', 'NC', 'RG', 'SP', 'DA', 'NC', 'SP', 'NC', 'PR', 'DA', 'NC', 'VMI', 'SP', 'AQ', 'NC', 'AQ', 'Fp']
Token count: 38, Tag count: 38
Tokens: ['El', 'portavoz', 'del', 'Consejo', 'Político', 'Municipal', 'de', 'IU', 'en', 'Santander', ',', 'Ernesto', 'Gómez', 'de', 'la', 'Hera', ',', 'explicó', 'hoy', 'en', 'conferencia', 'de', 'prensa', 'que', 'este', 'boletín', 'persigue', ',', 'más', 'que', 'informar', ',', 'abrir', 'espacios', 'de', 'debate', 'y', 'servir', 'como', '"', 'semilla', 'de', 'movilización', 'política', '"', 'en', 'la', 'ciudad', '.']
Tags: ['DA', 'NC', 'SP', 'NC', 'AQ', 'AQ', 'SP', 'NC', 'SP', 'NC', 'Fc', 'NC', 'NC', 'SP', 'DA', 'NC', 'Fc', 'VMI', 'RG', 'SP', 'NC', 'SP', 'NC', 'PR', 'DD', 'NC', 'VMI', 'Fc', 'RG', 'CS', 'VMN', 'Fc', 'VMN', 'NC', 'SP', 'NC', 'CC', 'VMN', 'CS', 'Fe', 'NC', 'SP', 'NC', 'AQ', 'Fe', 'SP', 'DA', 'NC', 'Fp']
Token count: 49, Tag count: 49
Tokens: ['Gómez', 'de', 'la', 'Hera', 'comentó', 'que', 'los', 'contenidos', 'de', 'la', 'publicación', '"', 'Ciudad', '"', 'responden', 'al', 'programa', 'que', 'él', 'encabezó', 'como', 'candidato', 'en', 'las', 'elecciones', 'municipales', 'de', '1999', ',', 'en', 'las', 'que', 'IU', 'perdió', 'su', 'representación', 'en', 'el', 'Ayuntamiento', ',', 'y', 'confió', 'en', 'que', 'su', 'difusión', 'sea', 'más', 'amplia', 'una', 'vez', 'que', 'la', 'coalición', 'resuelva', 'su', 'crisis', 'económica', '.']
Tags: ['NC', 'SP', 'DA', 'NC', 'VMI', 'CS', 'DA', 'AQ', 'SP', 'DA', 'NC', 'Fe', 'NC', 'Fe', 'VMI', 'SP', 'NC', 'PR', 'PP', 'VMI', 'CS', 'NC', 'SP', 'DA', 'NC', 'AQ', 'SP', 'Z', 'Fc', 'SP', 'DA', 'PR', 'NC', 'VMI', 'DP', 'NC', 'SP', 'DA', 'NC', 'Fc', 'CC', 'VMI', 'SP', 'CS', 'DP', 'NC', 'VSS', 'RG', 'AQ', 'DI', 'NC', 'PR', 'DA', 'NC', 'VMS', 'DP', 'NC', 'AQ', 'Fp']
Token count: 59, Tag count: 59
Tokens: ['Por', 'el', 'momento', ',', '"', 'Ciudad', '"', 'será', 'repartida', 'de', 'forma', 'preferencial', 'en', 'los', 'barrios', 'con', 'más', 'electores', 'de', 'IU', ',', 'entre', 'los', 'que', 'el', 'portavoz', 'municipal', 'de', 'la', 'coalición', 'citó', 'la', 'ladera', 'norte', 'de', 'la', 'calle', 'Alta', ',', 'Cazoña', 'y', 'la', 'zona', 'comprendida', 'entre', 'las', 'calles', 'Castilla', 'y', 'Hermida', '.']
Tags: ['SP', 'DA', 'NC', 'Fc', 'Fe', 'NC', 'Fe', 'VSI', 'AQ', 'SP', 'NC', 'AQ', 'SP', 'DA', 'NC', 'SP', 'RG', 'NC', 'SP', 'NC', 'Fc', 'SP', 'DA', 'PR', 'DA', 'NC', 'AQ', 'SP', 'DA', 'NC', 'VMI', 'DA', 'AQ', 'NC', 'SP', 'DA', 'NC', 'AQ', 'Fc', 'NC', 'CC', 'DA', 'NC', 'VMP', 'SP', 'DA', 'NC', 'NC', 'CC', 'NC', 'Fp']
Token count: 51, Tag count: 51
Tokens: ['"', 'Ciudad', '"', ',', 'un', 'boletín', 'compuesto', 'por', 'dos', 'folios', 'plegados', 'por', 'la', 'mitad', ',', 'sale', 'a', 'la', 'calle', 'en', 'su', 'primer', 'número', 'criticando', 'las', 'obras', 'de', 'construcción', 'de', 'los', 'aparcamientos', 'subterráneos', ',', 'las', 'molestias', 'que', 'están', 'causando', 'a', 'los', 'vecinos', 'y', 'el', 'modo', 'en', 'que', 'se', 'explotarán', '.']
Tags: ['Fe', 'NC', 'Fe', 'Fc', 'DI', 'NC', 'AQ', 'SP', 'DN', 'NC', 'NC', 'SP', 'DA', 'NC', 'Fc', 'VMI', 'SP', 'DA', 'NC', 'SP', 'DP', 'AO', 'NC', 'VMG', 'DA', 'NC', 'SP', 'NC', 'SP', 'DA', 'NC', 'AQ', 'Fc', 'DA', 'NC', 'PR', 'VMI', 'VMG', 'SP', 'DA', 'NC', 'CC', 'DA', 'NC', 'SP', 'PR', 'P0', 'VMI', 'Fp']
Token count: 49, Tag count: 49
Tokens: ['Gómez', 'de', 'la', 'Hera', 'considera', 'que', '"', 'la', 'política', 'del', 'PP', 'de', 'hacer', 'dinero', 'absolutamente', 'de', 'todo', 'y', 'de', 'que', 'ese', 'dinero', 'se', 'distribuya', 'sobre', 'todo', 'para', 'sus', 'amigos', '"', 'ha', 'convertido', 'Santander', 'en', '"', 'una', 'trampa', 'llena', 'de', 'agujeros', ',', 'cuando', 'todas', 'las', 'ciudades', 'de', 'Europa', 'occidental', 'están', 'tratando', 'de', 'hacer', 'sus', 'cascos', 'urbanos', 'más', 'habitables', '"', '.']
Tags: ['VMI', 'SP', 'DA', 'NC', 'VMI', 'CS', 'Fe', 'DA', 'NC', 'SP', 'NC', 'SP', 'VMN', 'NC', 'RG', 'SP', 'PI', 'CC', 'SP', 'CS', 'DD', 'NC', 'P0', 'VMS', 'SP', 'PI', 'SP', 'DP', 'NC', 'Fe', 'VAI', 'VMP', 'NC', 'SP', 'Fe', 'DI', 'NC', 'AQ', 'SP', 'NC', 'Fc', 'CS', 'DI', 'DA', 'NC', 'SP', 'NC', 'AQ', 'VMI', 'VMG', 'SP', 'VMN', 'DP', 'NC', 'AQ', 'RG', 'AQ', 'Fe', 'Fp']
Token count: 59, Tag count: 59
Tokens: ['A', 'su', 'juicio', ',', 'las', 'razones', 'del', 'Ayuntamiento', 'para', 'llevar', 'adelante', 'estas', 'obras', 'son', '"', 'crematísticas', ',', 'mercenarias', ',', 'pero', 'no', 'tienen', 'nada', 'que', 'ver', 'con', 'el', 'futuro', 'de', 'Santander', 'ni', 'con', 'el', 'bienestar', 'de', 'la', 'inmensa', 'mayoría', 'de', 'su', 'población', '"', '.']
Tags: ['SP', 'DP', 'NC', 'Fc', 'DA', 'NC', 'SP', 'NC', 'SP', 'VMN', 'RG', 'DD', 'NC', 'VSI', 'Fe', 'AQ', 'Fc', 'AQ', 'Fc', 'CC', 'RN', 'VMI', 'PI', 'PR', 'VMN', 'SP', 'DA', 'NC', 'SP', 'NC', 'CC', 'SP', 'DA', 'NC', 'SP', 'DA', 'AQ', 'NC', 'SP', 'DP', 'NC', 'Fe', 'Fp']
Token count: 43, Tag count: 43
Tokens: ['EFE', '-', 'Cantabria', 'Fráncfort', '(', 'RFA', ')', ',', '25', 'may', '(', 'EFECOM', ')', '.']
Tags: ['NC', 'Fg', 'VMI', 'AQ', 'Fpa', 'NP', 'Fpt', 'Fc', 'Z', 'NC', 'Fpa', 'NP', 'Fpt', 'Fp']
Token count: 14, Tag count: 14
Tokens: ['-']
Tags: ['Fg']
Token count: 1, Tag count: 1
Tokens: ['El', 'grupo', 'alemán', 'de', 'telefonía', 'Deutsche', 'Telekom', 'ha', 'adquirido', 'el', '50', 'por', 'ciento', 'restante', 'del', 'operador', 'suizo', 'de', 'telecomunicaciones', 'Multilink', ',', 'que', 'hasta', 'ahora', 'poseía', 'France', 'Telecom', ',', 'y', 'se', 'ha', 'convertido', 'en', 'su', 'único', 'propietario', ',', 'informó', 'hoy,', 'jueves', ',', 'la', 'compañía', 'germana', '.']
Tags: ['DA', 'NC', 'AQ', 'SP', 'NC', 'AQ', 'NC', 'VAI', 'VMP', 'DA', 'Z', 'SP', 'DN', 'NC', 'SP', 'NC', 'AQ', 'SP', 'NC', 'AQ', 'Fc', 'CS', 'SP', 'RG', 'VMI', 'AQ', 'AQ', 'Fc', 'CC', 'P0', 'VAI', 'VMP', 'SP', 'DP', 'AQ', 'NC', 'Fc', 'VMI', 'AQ', 'NC', 'Fc', 'DA', 'NC', 'AQ', 'Fp']
Token count: 45, Tag count: 45
"""
