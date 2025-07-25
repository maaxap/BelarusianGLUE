#!/usr/bin/python3
# coding: utf-8

# Imports

import json
import hashlib

import streamlit as st
import requests

# Constants

SCRIPT_URL = "https://script.google.com/macros/s/<anonymized>/exec"

DATASETS = ["besls", "belacola", "bewic", "bewsc", "bertewd"]
DATASET_TITLES = {
    "besls": "BeSLS",
    "belacola": "BelaCoLA",
    "bewic": "BeWiC",
    "bewsc": "BeWSC",
    "bertewd": "BeRTE-WD",
}
DATASET_OPTIONS = {
    "besls": ["негатыўны", "пазітыўны"],
    "belacola": ["непрымальны", "прымальны"],
    "bewic": ["розныя", "аднолькавыя"],
    "bewsc": ["не", "так"],
    "bertewd": ["не", "так"],
}

PAGE = "page"
INDEX = "index"
INTRO = "intro"
SELFCHECK = "selfcheck"
READY = "ready"
RUNNING = "running"
SUBMITTED = "submitted"
COMMENTS = "_comments"

SELFCHECK_MARKS = [
    '<span style="color: red; font-weight: bold; font-size: 2em">✗</span> ',
    '<span style="color: green; font-weight: bold; font-size: 2em">✔</span> '
]

SECTION_SIZE = 20

CUSTOM_STYLE = """<style>
html { font-size: 120%; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0%; position: fixed; }
div[data-testid="stDecoration"] { visibility: hidden; height: 0%; position: fixed; }
div[data-testid="stStatusWidget"] { visibility: hidden; height: 0%; position: fixed; }
#MainMenu { visibility: hidden; height: 0%; }
header { visibility: hidden; height: 0%; }
footer { visibility: hidden; height: 0%; }
</style>"""

# Functions

idx_to_path = lambda idx: str(idx) + hashlib.md5(str(idx).encode("utf-8")).hexdigest()[:15]
path_to_idx = lambda path: int(path[0]) - 1

def read_data():
    data = {}
    for filename in DATASETS:
        data[filename] = {}

        with open("instructions/intro/%s" % filename) as f:
            data[filename][INTRO] = f.read().strip()

        with open("instructions/selfcheck/%s" % filename) as f:
            data[filename][SELFCHECK] = [line.strip().split("\t") for line in f]
        assert len(data[filename][SELFCHECK]) == 5 and {len(row) for row in data[filename][SELFCHECK]} == {2}

        tab_replacement = "<br/>" if filename == "bewic" else "<br/>⇒ "
        with open("labeling/sentences/%s" % filename) as f:
            labeling_data = [line.strip().replace("\t", tab_replacement) for line in f]
        assert len(labeling_data) == 100
        for i in range(5):
            data[filename][idx_to_path(i+1)] = labeling_data[i*SECTION_SIZE:(i+1)*SECTION_SIZE]
    return data

@st.fragment
def make_self_check_items(data, filename):
    self_check_items = []
    for i, (_, instance) in enumerate(data[filename][SELFCHECK]):
        k = "selfcheck" + str(i)
        col1, col2 = st.columns((1, 3), vertical_alignment="center")
        self_check_items.append([
            col1.radio(label=k, options=DATASET_OPTIONS[filename], index=None, key=k, label_visibility="collapsed"),
            col2.empty()
        ])
        self_check_items[-1][1].markdown(instance, unsafe_allow_html=True)
        st.divider()

    # Self-check button
    if st.button("Праверыць сябе"):
        requires_self_check = False
        for i, (label, instance) in enumerate(data[filename][SELFCHECK]):
            answer = self_check_items[i][0]
            if answer is None:
                continue
            requires_self_check = True
            prefix = SELFCHECK_MARKS[answer == DATASET_OPTIONS[filename][int(label)]]
            self_check_items[i][1].markdown(prefix + instance, unsafe_allow_html=True)
        if not requires_self_check:
            st.warning("Спачатку выберыце хаця б адзін адказ.", icon="⚠️")

@st.fragment
def make_labeled_items(data, filename, user_id):
    labeled_items = {}
    for i, instance in enumerate(data[filename][user_id]):
        k = filename + str(SECTION_SIZE * path_to_idx(user_id) + i)
        col1, col2 = st.columns((1, 3), vertical_alignment="center")
        labeled_items[k] = col1.radio(
            label=k, options=DATASET_OPTIONS[filename], index=None,
            key=k, label_visibility="collapsed", disabled=st.session_state[filename][SUBMITTED]
        )
        col2.markdown(("%s. " % (i + 1)) + instance, unsafe_allow_html=True)
        st.divider()
    k_comm = filename + COMMENTS
    @st.fragment
    def make_comment_box():
        return st.text_area(
            label=k_comm, placeholder="Калі ёсць заўвагі ці каментары, можна іх пакінуць тут",
            label_visibility="collapsed", disabled=st.session_state[filename][SUBMITTED]
        )
    labeled_items[k_comm] = make_comment_box()

    st.session_state[filename][READY] = st.checkbox("Адказы гатовыя")

    def is_disabled():
        return (not st.session_state[filename][READY]) or st.session_state[filename][RUNNING] or st.session_state[filename][SUBMITTED]

    def start_running():
        if not any(labeled_items[k] is None for k in labeled_items if not k.endswith(COMMENTS)):
            st.session_state[filename][RUNNING] = True

    def format_answers():
        format_value = lambda k, v: v if k.endswith(COMMENTS) else DATASET_OPTIONS[filename].index(v)
        return {
            "sheet_id": user_id,
            "data": [
                {"instance_id": k, "instance_label": format_value(k, v)}
                for k, v in labeled_items.items()
            ]
        }

    # Submit button
    if st.button("Даслаць адказы", on_click=start_running, disabled=is_disabled()):
        if st.session_state[filename][RUNNING]:
            answers = format_answers()
            response = requests.post(SCRIPT_URL, json=answers)
            if response.status_code == 200:
                st.session_state[filename][SUBMITTED] = True
                st.rerun()
            else:
                st.error(f"Не атрымалася даслаць адказы: HTTP {str(response.status_code)}.\nМожна паспрабаваць яшчэ раз, а калі памылка не знікне, варта паведаміць пра яе арганізатарам. Даруйце нязручнасці!")
                st.download_button(
                    label="Захаваць адказы лакальна",
                    data=json.dumps(answers, ensure_ascii=False, indent=2),
                    file_name="answers_" + user_id,
                    mime="application/json"
                )
            st.session_state[filename][RUNNING] = False
        else:
            st.warning("Перш чым дасылаць адказы, калі ласка, завяршыце разметку ўсіх 20 прыкладаў. Дзякуй!", icon="⚠️")

    if st.session_state[filename][SUBMITTED]:
        st.success("Адказы паспяхова дасланы. Дзякуй!")

def labeling(data, filename, user_id):
    assert filename in data and INTRO in data[filename] and SELFCHECK in data[filename] and user_id in data[filename]

    # Instruction
    st.markdown(data[filename][INTRO], unsafe_allow_html=True)

    # Self-check instances
    st.divider()
    st.write("Яшчэ 5 прыкладаў, якія можна размеціць і потым праверыць сябе:")
    st.divider()
    make_self_check_items(data, filename)

    # Instances for labeling
    st.divider()
    st.write("Вось 20 прыкладаў для самастойнай разметкі. Перш чым дасылаць адказы, калі ласка, завяршыце разметку ўсіх прыкладаў:")
    st.divider()
    make_labeled_items(data, filename, user_id)

def main(data):
    # Configure page appearance
    st.set_page_config(
        page_title="Labeling UI for the human baseline of Belarusian GLUE",
        page_icon=None,
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None
    )

    st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

    # Initialize session state
    if PAGE not in st.session_state:
        st.session_state.page = INDEX
    for dataset in DATASETS:
        if dataset not in st.session_state:
            st.session_state[dataset] = {
                READY: False,
                RUNNING: False,
                SUBMITTED: False
            }

    # Using an approximately balanced Latin square to reduce order effects
    DATASET_PERMUTATIONS = {
        idx_to_path(1): (3, 4, 0, 1, 2),
        idx_to_path(2): (2, 1, 4, 3, 0),
        idx_to_path(3): (4, 2, 3, 0, 1),
        idx_to_path(4): (0, 3, 1, 2, 4),
        idx_to_path(5): (1, 0, 2, 4, 3),
    }

    # Verify labeler ID
    if "id" not in st.query_params or st.query_params["id"] not in DATASET_PERMUTATIONS:
        st.markdown("Памылковы ідэнтыфікатар. Калі ласка, праверце параметр `id` у адрасе старонкі.")
        return

    # Render the appropriate page
    if st.session_state.page == INDEX:
        st.title("Галоўная старонка")
        st.write("Мы працуем над беларускамоўным рэсурсам, які дазволіць ацэньваць якасць працы вялікіх моўных мадэляў (LLM), такіх як ChatGPT. Рэсурс складаецца з тэставых заданняў, у якіх правяраюцца розныя аспекты разумення мовы, і трэба ўпэўніцца, што эталонныя адказы супадаюць з інтуіцыяй носьбітаў мовы. Для разметкі прыкладаў нам неабходна дапамога экспертаў.")
        st.write("Запрашаем размеціць па 20 прыкладаў з 5 катэгорый:")
        for i in DATASET_PERMUTATIONS[st.query_params["id"]]:
            dataset = DATASETS[i]
            if st.button(DATASET_TITLES[dataset]):
                st.session_state.page = dataset
                st.rerun()
        st.write("Падрабязныя інструкцыі да кожнай катэгорыі глядзіце па спасылках.")
        st.write("Вы можаце пераходзіць да катэгорый у любым парадку. Зусім не абавязкова размячаць адразу ўсе 100 прыкладаў – каб не занадта стамляцца, рабіце перапынкі паміж катэгорыямі.")
        st.write("Выконваць заданні можна як на звычайным камп’ютары, так і на мабільнай прыладзе; у апошнім выпадку, у залежнасці ад мадэлі, могуць час ад часу ўзнікаць тэхнічныя праблемы.")
        st.write("Дзякуй за ўдзел у даследаванні!")
    else:
        assert st.session_state.page in DATASET_TITLES
        if st.button("На галоўную старонку", key="back_top"):
            st.session_state.page = INDEX
            st.rerun()
        st.title(DATASET_TITLES[st.session_state.page])
        _ = labeling(data, st.session_state.page, st.query_params["id"])
        st.divider()
        if st.button("На галоўную старонку", key="back_bottom"):
            st.session_state.page = INDEX
            st.rerun()

if __name__ == "__main__":
    _ = main(read_data())
