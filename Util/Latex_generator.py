import pandas as pd
import os

pd.options.display.max_colwidth = 100

def generate_latex_table(data_file, labels, tab_name, caption, header):

    with open(tab_name + ".tex", "w") as f:
        df = pd.read_excel(data_file, sheet_name="Summary")
        df_restricted = df[labels]
        f.write(df_restricted.to_latex(index=False, escape=False,
                                       label="tab:{}".format(tab_name),
                                       caption= caption,
                                       header = header))


def generate_comparison_latex_table():
    labels = ["Domain", "Neg precision A", "Neg recall A", "Overall precision A", "Overall recall A",
              "Neg precision B", "Neg recall B", "Overall precision B", "Overall recall B"]
    header = ["Domain", "$P_{\\eff^{-}}$", "$R_{\\eff^{-}}$", "$P$", "$R$",
              "$P_{\\eff^{-}}$", "$R_{\\eff^{-}}$", "$P$", "$R$"]
    caption = "For each domain:statistics on final metrics of the last instance grouped by " \
              "negative effects."
    tab_name = "comparison_summary_uncertain"
    file_path = os.path.join("comparison_summary_uncertain.xlsx")

    generate_latex_table(file_path, labels, tab_name, caption, header)


def generate_comparison_latex_table_fama():
    labels = ["Domain", "Tot time", "Overall precision", "Overall recall", "FAMA tot time",
              "FAMA precision", "FAMA recall", "Delta act"]
    header = ["Domain", "$t$", "$P$", "$R$", "$t$", "$P$", "$R$", "$\delta_{A}$"]
    caption = "Comparison among OLAM and FAMA with full observability. FAMA is run with all plan traces " \
              "provided in \protect\cite{aineto_AIJ2019}. MODEL WITH UNCERTAIN NEGATIVE EFFECTS AND STRIPS ASSUMPTION."
    tab_name = "comparison_fama"
    file_path = os.path.join("comparison_fama.xlsx")

    generate_latex_table(file_path, labels, tab_name, caption, header)


def generate_summary_latex_table():
    # labels = ["Domain", "Instances", "Precs precision",  "Precs recall","Pos precision", "Pos recall",
    #           "Neg precision", "Neg recall", "Overall precision", "Overall recall"]
    labels = ["Domain", "Instances", "Precs precision",  "Precs recall","Pos precision", "Pos recall",
              "Neg precision", "Neg recall", "Average precision", "Average recall"]
    header = ["Domain", "$I$", "$P_{\\prec}$", "$R_{\\prec}$", "$P_{\\eff^{+}}$", "$R_{\\eff^{+}}$", "$P_{\\eff^{-}}$",
              "$R_{\\eff^{-}}$", "$P$", "$R$"]
    caption = "For each domain:statistics on final metrics of the last instance grouped by " \
              "preconditions, positive effects and negative ones."
    tab_name = "overall_summary_certain_nostripsass"

    folder = "../Analysis/IJCAI_Results/Results_certain_NOnegeff_assumption"
    file_path = os.path.join(folder, "overall_summary.xlsx")

    generate_latex_table(file_path, labels, tab_name, caption, header)


def generate_domain_objects_table():

    header = ["Domain", "Objects"]
    caption = "For each domain, problem objects of all problems in the generated set."
    tab_name = "all_problem_objects"

    df = pd.DataFrame({
        "Domain":[],
        "Objects":[]
    })
    # df.set_index('Domain', inplace=True)

    domain_dataframes = [name for name in os.listdir(os.path.join("..", "Analysis", "Results_cert"))
                         if not name.startswith("overall")]

    for domain_dataframe in domain_dataframes:
        domain = domain_dataframe.split("_")[0]
        df_domain = pd.read_excel(os.path.join("..", "Analysis", "Results_cert", domain_dataframe),
                                  sheet_name="Objects")
        domain_obj_types = [key.strip().lower() for key in list(df_domain) if key.strip().lower() != "total objs"]

        for i, row in df_domain.iterrows():
            problem_objs = []
            for k in domain_obj_types:
                problem_objs.append("{} {}".format(k,row["\t" + k]))

            eval = {
                "Domain":domain,
                "Objects":", ".join(problem_objs)
            }


            df = df.append(eval, ignore_index=True)




    with open(tab_name + ".tex", "w") as f:
        f.write(df.to_latex(index=False,
                                       label="tab:{}".format(tab_name),
                                       caption= caption,
                                       header = header))



if __name__ == "__main__":

    generate_summary_latex_table()
    #
    # generate_domain_objects_table()
