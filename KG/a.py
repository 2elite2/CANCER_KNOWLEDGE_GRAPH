import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt


def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]


def printToken(token):
    print(token.text, "->", token.dep_)


def appendChunk(original, chunk):
    return original + ' ' + chunk


def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)


def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)


def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(
                    subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(
                    objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''

    print(subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())


def processSentence(sentence):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)


def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='seagreen', alpha=0.9,
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()


if __name__ == "__main__":

    text = "Leukaemia is a cancer disease in which the symptoms as pale skin tone, fever, easy bleeding, bone or joint pain, swelling in lymph nodes, and pain under ribs on the left side. Individuals who was suffering from leukaemia cancer might have these symptoms visible very clearly depending upon the stage of cancer he/she is on. One can get the diagnosis done for this kind of cancer by a physical examination, bone marrow biopsy, MRI CT-SCAN, chest X-ray. The various kinds of treatments for leukaemia are chemotherapy radiation therapy immunotherapy and a hematopoietic cell transplant. Also, as per research the food which should be avoided in this disease are the foods that are high in fibre or sugar, greasy, fatty, or fried foods, very hot or very cold foods, milk products and alcohol. Fruits and vegetables cold greens and legumes high protein foods and cruciferous vegetables have proved to be very beneficial in the recovery of this disease."

    sentences = getSentences(text)
    nlp_model = spacy.load('en_core_web_sm')

    triples = []
    print(text)
    for sentence in sentences:
        triples.append(processSentence(sentence))

    printGraph(triples)
    
# python -m spacy download en_core_web_sm