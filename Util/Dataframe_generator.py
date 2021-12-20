import argparse
import datetime
import re
from collections import defaultdict
from pathlib import Path
import shutil

import pandas as pd
import os

from scipy import interpolate

import Configuration
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def preprocess(data):

    filtered = [str.replace("\t\t", " ").replace("|","").split() for str in data
                if str.strip() != "" and all([el.isnumeric()
                for el in str.replace("\t\t", " ").replace(".","").replace("|","").split()])]

    filtered = np.array(filtered)

    times = [float(str) for str in filtered[:, 0]]
    iterations = [float(str) for str in filtered[:, 1]]
    prec_recall = [float(str) for str in filtered[:, 2]]
    prec_precision = [float(str) for str in filtered[:, 3]]
    eff_recall = [float(str) for str in filtered[:, 4]]
    eff_precision = [float(str) for str in filtered[:, 5]]
    overall_recall = [float(str) for str in filtered[:, 6]]
    overall_precision = [float(str) for str in filtered[:, 7]]

    return times, iterations, prec_recall, prec_precision, eff_recall, eff_precision, overall_recall, overall_precision



def generate_domain_dataframes(uncert_neg_effects = False):

    # TEST_DIR = "Tests"
    # RESULTS_DIR = "Results"

    if uncert_neg_effects:
        RESULTS_DIR = "Results_uncert_neg"
    else:
        RESULTS_DIR = "Results_cert"

    # RESULTS_DIR = os.path.join("Analysis", RESULTS_DIR)
    RESULTS_DIR = os.path.join("/".join(Configuration.ROOT_TEST_DIR.split('/')[:-1]), RESULTS_DIR)

    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    os.mkdir(RESULTS_DIR)


    # for domain in os.listdir(os.path.join("Analysis", TEST_DIR)):
    for domain in [d for d in os.listdir(Configuration.ROOT_TEST_DIR) if not d.lower().startswith("results")]:

        checked_log = False
        checked_facts = False
        checked_domain = False
        extended_dataframe = False

        # instances_dir = "{}{}".format(Configuration.ROOT_TEST_DIR, domain)
        instances_dir = os.path.join(Configuration.ROOT_TEST_DIR, domain)

        df = pd.DataFrame({'Instance': [],
                           'Ground actions': [],
                            'Computation time': [],
                           'Executed actions': [],
                            'Success actions': [],
                           'Failed actions': [],
                           'Real precs': [],
                           'Learn precs': [],
                           'Real pos': [],
                           'Learn pos': [],
                           'Real neg': [],
                           'Learn neg': [],
                           'Ins_pre': [],
                           'Del_pre': [],
                           'Ins_pos': [],
                           'Del_pos': [],
                           'Ins_neg': [],
                           'Del_neg': [],
                           'Precs recall': [],
                           'Pos recall': [],
                           'Neg recall': [],
                           'Precs precision': [],
                           'Pos precision': [],
                           'Neg precision': [],
                           'Overall recall': [],
                           'Overall precision': []
                           })

        all_benchmark_dirs = []
        for benchmark_subdir, dirs, files in os.walk(instances_dir):
            all_benchmark_dirs = dirs
            break

        max_goal = 1
        for dir in sorted(all_benchmark_dirs, key = lambda x: int(x.split("_")[0])):
            for subdir, dirs, files in os.walk("{}/{}".format(instances_dir, dir)):

                for file in files:

                    if file.endswith("log"):
                        instance_name = file[:file.index("_" + file.split("_")[-1])].strip()
                        benchmark_dir = dir
                        plan_goal = 0
                        total_actions = None
                        executed_actions = 0
                        failed_actions = 0
                        success_actions = 0

                        # precs_precision = None
                        # precs_recall = None
                        # eff_precision = None
                        # eff_recall = None
                        # overall_precision = None
                        # overall_recall = None
                        #
                        # computation_time = None


                        with open(os.path.join(subdir, file), "r") as f:
                            data = [el.replace("|","") for el in f.read().split('\n') if el.strip() != ""]

                            # computation_time = data[-1].split()[0]
                            # precs_recall = data[-1].split()[2]
                            # precs_precision = data[-1].split()[3]
                            # eff_recall = data[-1].split()[4]
                            # eff_precision = data[-1].split()[5]
                            # overall_recall = data[-1].split()[6]
                            # overall_precision = data[-1].split()[7]

                            all_metrics = [str.replace("\t\t", " ").replace("|", "").split() for str in data
                                        if str.strip() != "" and all([el.isnumeric()
                                                                      for el in str.replace("\t\t", " ").replace(".", "").replace(
                                                                          "|", "").split()])]

                            #################### Consider certain domain metrics ################
                            if not uncert_neg_effects:
                                all_metrics = all_metrics[:-1]
                            #####################################################################

                            computation_time = all_metrics[-1][0]
                            real_precs = all_metrics[-1][2]
                            learn_precs = all_metrics[-1][3]
                            real_pos = all_metrics[-1][4]
                            learn_pos = all_metrics[-1][5]
                            real_neg = all_metrics[-1][6]
                            learn_neg = all_metrics[-1][7]
                            ins_pre = all_metrics[-1][8]
                            del_pre = all_metrics[-1][9]
                            ins_pos = all_metrics[-1][10]
                            del_pos = all_metrics[-1][11]
                            ins_neg = all_metrics[-1][12]
                            del_neg = all_metrics[-1][13]
                            precs_recall = all_metrics[-1][14]
                            pos_recall = all_metrics[-1][15]
                            neg_recall = all_metrics[-1][16]
                            precs_precision = all_metrics[-1][17]
                            pos_precision = all_metrics[-1][18]
                            neg_precision = all_metrics[-1][19]
                            overall_recall = all_metrics[-1][20]
                            overall_precision = all_metrics[-1][21]
                            inst_objs = defaultdict(int)

                            for i in range(len(data)):
                                line = data[i]

                                if line.strip().lower().startswith("not successfully executed"):
                                    failed_actions += 1
                                    executed_actions += 1

                                elif line.strip().lower().startswith("successfully executed"):
                                    success_actions += 1
                                    executed_actions += 1

                                elif line.strip().lower().startswith("objects list"):
                                    for j in range(i+1, len(data)):
                                        if data[j].lower().strip().startswith("time") and ":" not in data[j].lower():
                                            break
                                        inst_objs[data[j].lower().split(":")[0]] = int(data[j].lower().split(":")[1])



                        # Total actions
                        with open(os.path.join(subdir, file), "r") as f:
                            data = f.read().split('\n')

                            for i in range(len(data)):
                                line = data[i]

                                if line.strip().find("Total actions") != -1:
                                    total_actions = line.split(":")[1].strip()
                                    break


                        checked_log = True


                # if len(files) > 0:
                if checked_log:
                    evaluate = {
                                'Instance': instance_name,
                                'Ground actions': int(total_actions),
                                'Computation time': float(computation_time),
                                'Executed actions': int(executed_actions),
                                'Success actions': int(success_actions),
                                'Failed actions': int(failed_actions),
                               'Real precs': int(real_precs),
                               'Learn precs': int(learn_precs),
                               'Real pos': int(real_pos),
                               'Learn pos': int(learn_pos),
                               'Real neg': int(real_neg),
                               'Learn neg': int(learn_neg),
                               'Ins_pre': int(ins_pre),
                               'Del_pre': int(del_pre),
                               'Ins_pos': int(ins_pos),
                               'Del_pos': int(del_pos),
                               'Ins_neg': int(ins_neg),
                               'Del_neg': int(del_neg),
                               'Precs recall': float(precs_recall),
                               'Pos recall': float(pos_recall),
                               'Neg recall': float(neg_recall),
                               'Precs precision': float(precs_precision),
                               'Pos precision': float(pos_precision),
                               'Neg precision': float(neg_precision),
                                'Overall recall': float(overall_recall),
                                'Overall precision': float(overall_precision)
                                }
                    checked_log = False

                    if not extended_dataframe:
                        for obj_type in inst_objs.keys():
                            extended_dataframe = True
                            df[obj_type] = -1
                            df["Total objs"] = -1

                    total_objs = 0
                    for obj_type in inst_objs.keys():
                        evaluate[obj_type] = inst_objs[obj_type]
                        total_objs += inst_objs[obj_type]

                    evaluate["Total objs"] = total_objs

                    df = df.append(evaluate, ignore_index=True)
                        # print(data)

        ordered_instances = None

        ordered_instances = [el.replace(".pddl", "") for el in
                             sorted(os.listdir(benchmark_subdir), key = lambda x: int(x.split("_")[0]))]

        df['Instance'] = pd.Categorical(df.Instance, categories=ordered_instances, ordered=True)

        df.sort_values("Instance", inplace=True)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(os.path.join(RESULTS_DIR,"{}_action_model_eval.xls".format(domain)))



        summary = ['Instance',  'Computation time', 'Executed actions', 'Overall recall', 'Overall precision']


        df.to_excel(writer, sheet_name='Total', index=False)
        df[summary].to_excel(writer, sheet_name='Summary', index=False)
        df[list(inst_objs.keys()) + ["Total objs"]].to_excel(writer, sheet_name='Objects', index=False)

        writer.close()

        # print("minimum number of objects of {} : {}".format(domain, min(df["Total objs"])))
        # print("maximum number of objects of {} : {}".format(domain, max(df["Total objs"])))

        # print("minimum number of ground actions of {} : {}".format(domain, min(df["Ground actions"])))
        # print("maximum number of ground actions of {} : {}".format(domain, max(df["Ground actions"])))


def generate_fama_average_results():

    all_domains_time = defaultdict(list)
    all_domains_recall = defaultdict(list)
    all_domains_precision = defaultdict(list)

    for execution_result in os.listdir("Results_FAMA"):

        df_path = os.path.join("Results_FAMA", execution_result)

        df = pd.read_excel(df_path)

        for index, row in df.iterrows():

            domain = row['domain']
            time = float(row['time'])
            recall = float(row['recall'])
            precision = float(row['precision'])

            all_domains_time[domain].append(time)
            all_domains_recall[domain].append(recall)
            all_domains_precision[domain].append(precision)





    # Create summary dataframe over all domains
    df = pd.DataFrame({'Domain': [],
                       'Avg time': [],
                       'Avg recall': [],
                       'Avg precision': []
                       })



    for domain in all_domains_time.keys():

        results_exec = len(all_domains_time[domain])
        # avg_time = sum(all_domains_time[domain])/results_exec
        # avg_precision = sum(all_domains_precision[domain])/results_exec
        # avg_recall = sum(all_domains_recall[domain])/results_exec
        med_time = sorted(all_domains_time[domain])[2]
        med_precision = sorted(all_domains_precision[domain])[2]
        med_recall = sorted(all_domains_recall[domain])[2]

        evaluate = {'Domain': domain,
                       'Avg time': float(med_time),
                       'Avg recall': float(med_recall),
                       'Avg precision': float(med_precision)
        }

        df = df.append(evaluate, ignore_index=True)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('Results_FAMA/average_results.xlsx', engine='xlsxwriter')

    df.to_excel(writer, index=False, float_format = "%0.2f")

    writer.close()


def generate_domain_summary(uncert_neg_effects = False):

    # RESULTS_DIR = ""

    if uncert_neg_effects:
        RESULTS_DIR = "Results_uncert_neg"
    else:
        RESULTS_DIR = "Results_cert"

    # RESULTS_DIR = os.path.join("Analysis", RESULTS_DIR)
    RESULTS_DIR = os.path.join("/".join(Configuration.ROOT_TEST_DIR.split('/')[:-1]), RESULTS_DIR)

    if os.path.exists(os.path.join(RESULTS_DIR, "overall_summary.xls")):
        os.remove(os.path.join(RESULTS_DIR, "overall_summary.xls"))

    all_domains = os.listdir(RESULTS_DIR)

    all_domains_total_objs = defaultdict(list)
    all_domains_executed_actions = defaultdict(list)
    all_domains_success_actions = defaultdict(list)
    all_domains_failed_actions = defaultdict(list)
    all_domains_time = defaultdict(list)
    all_domains_ground_actions = defaultdict(list)

    all_domains_recall_prec = defaultdict(list)
    all_domains_precision_prec = defaultdict(list)

    all_domains_recall_pos = defaultdict(list)
    all_domains_precision_pos = defaultdict(list)

    all_domains_recall_neg = defaultdict(list)
    all_domains_precision_neg = defaultdict(list)

    all_domains_recall = defaultdict(list)
    all_domains_precision = defaultdict(list)

    all_domains_correctness = defaultdict(list)
    all_domains_integrity = defaultdict(list)

    for domain in [el for el in sorted(all_domains) if not el.startswith("overall")]:

        df_path = os.path.join(RESULTS_DIR, domain)

        df = pd.read_excel(df_path, sheet_name="Total")

        domain_total_objs = []
        domain_executed_actions = []
        domain_success_actions = []
        domain_failed_actions = []
        domain_ground_actions = []
        domain_time = []
        domain_recall_prec = []
        domain_precision_prec = []
        domain_recall_pos = []
        domain_precision_pos = []
        domain_recall_neg = []
        domain_precision_neg = []
        domain_recall = []
        domain_precision = []
        domain_correctness = []
        domain_integrity = []

        max_recall = 0
        max_precision = 0

        for index, row in df.iterrows():

            if row['Overall recall'] > max_recall and row['Overall precision'] > max_precision:
            # if (row['Overall recall'] > max_recall and row['Overall precision'] > max_precision)\
            #         or Configuration.RANDOM_WALK:

                max_recall = float(row['Overall recall'])
                max_precision = float(row['Overall precision'])

                max_recall_precs = float(row['Precs recall'])
                max_precision_precs = float(row['Precs precision'])

                max_recall_pos = float(row['Pos recall'])
                max_precision_pos = float(row['Pos precision'])

                max_recall_neg = float(row['Neg recall'])
                max_precision_neg = float(row['Neg precision'])

                domain_executed_actions.append(row['Executed actions'])
                domain_success_actions.append(row['Success actions'])
                domain_failed_actions.append(row['Failed actions'])
                domain_ground_actions.append(row['Ground actions'])
                domain_time.append(row['Computation time'])
                domain_total_objs.append(row['Total objs'])

                domain_recall_prec.append(row['Precs recall'])
                domain_precision_prec.append(row['Precs precision'])
                domain_recall_pos.append(row['Pos recall'])
                domain_precision_pos.append(row['Pos precision'])
                domain_recall_neg.append(row['Neg recall'])
                domain_precision_neg.append(row['Neg precision'])
                domain_recall.append(row['Overall recall'])
                domain_precision.append(row['Overall precision'])

                # Compute correctness
                size_m = row['Learn precs'] + row['Learn pos'] + row['Learn neg']
                del_pos = row['Del_pos']
                ins_pre = row['Ins_pre']
                ins_neg = row['Ins_neg']
                correctness = (size_m - del_pos)/(size_m + ins_pre + ins_neg)
                domain_correctness.append(correctness)

                # Compute integrity
                size_m = row['Learn precs'] + row['Learn pos'] + row['Learn neg']
                del_pre = row['Del_pre']
                del_neg = row['Del_neg']
                ins_pos = row['Ins_pos']
                integrity = (size_m - del_pre - del_neg)/(size_m + ins_pos)
                domain_integrity.append(integrity)


            elif row['Overall recall'] > max_recall:
            # elif row['Overall recall'] > max_recall or Configuration.RANDOM_WALK:

                max_recall = float(row['Overall recall'])

                domain_executed_actions.append(row['Executed actions'])
                domain_success_actions.append(row['Success actions'])
                domain_failed_actions.append(row['Failed actions'])
                domain_ground_actions.append(row['Ground actions'])
                domain_time.append(row['Computation time'])
                domain_total_objs.append(row['Total objs'])

                domain_recall_prec.append(row['Precs recall'])
                domain_precision_prec.append(row['Precs precision'])
                domain_recall_pos.append(row['Pos recall'])
                domain_precision_pos.append(row['Pos precision'])
                domain_recall_neg.append(row['Neg recall'])
                domain_precision_neg.append(row['Neg precision'])
                domain_recall.append(row['Overall recall'])
                domain_precision.append(row['Overall precision'])

                # Compute correctness
                size_m = row['Learn precs'] + row['Learn pos'] + row['Learn neg']
                del_pos = row['Del_pos']
                ins_pre = row['Ins_pre']
                ins_neg = row['Ins_neg']
                correctness = (size_m - del_pos)/(size_m + ins_pre + ins_neg)
                domain_correctness.append(correctness)

                # Compute integrity
                size_m = row['Learn precs'] + row['Learn pos'] + row['Learn neg']
                del_pre = row['Del_pre']
                del_neg = row['Del_neg']
                ins_pos = row['Ins_pos']
                integrity = (size_m - del_pre - del_neg)/(size_m + ins_pos)
                domain_integrity.append(integrity)

            elif row['Overall precision'] > max_precision:
            # elif row['Overall precision'] > max_precision or Configuration.RANDOM_WALK:

                max_precision = float(row['Overall precision'])

                domain_executed_actions.append(row['Executed actions'])
                domain_success_actions.append(row['Success actions'])
                domain_failed_actions.append(row['Failed actions'])
                domain_ground_actions.append(row['Ground actions'])
                domain_time.append(row['Computation time'])
                domain_total_objs.append(row['Total objs'])

                domain_recall_prec.append(row['Precs recall'])
                domain_precision_prec.append(row['Precs precision'])
                domain_recall_pos.append(row['Pos recall'])
                domain_precision_pos.append(row['Pos precision'])
                domain_recall_neg.append(row['Neg recall'])
                domain_precision_neg.append(row['Neg precision'])
                domain_recall.append(row['Overall recall'])
                domain_precision.append(row['Overall precision'])

                # Compute correctness
                size_m = row['Learn precs'] + row['Learn pos'] + row['Learn neg']
                del_pos = row['Del_pos']
                ins_pre = row['Ins_pre']
                ins_neg = row['Ins_neg']
                correctness = (size_m - del_pos)/(size_m + ins_pre + ins_neg)
                domain_correctness.append(correctness)

                # Compute integrity
                size_m = row['Learn precs'] + row['Learn pos'] + row['Learn neg']
                del_pre = row['Del_pre']
                del_neg = row['Del_neg']
                ins_pos = row['Ins_pos']
                integrity = (size_m - del_pre - del_neg)/(size_m + ins_pos)
                domain_integrity.append(integrity)


        all_domains_executed_actions[domain].extend(domain_executed_actions)
        all_domains_success_actions[domain].extend(domain_success_actions)
        all_domains_failed_actions[domain].extend(domain_failed_actions)
        all_domains_total_objs[domain].extend(domain_total_objs)
        all_domains_ground_actions[domain].extend(domain_ground_actions)
        all_domains_time[domain].extend(domain_time)

        all_domains_recall_prec[domain].extend(domain_recall_prec)
        all_domains_precision_prec[domain].extend(domain_precision_prec)

        all_domains_recall_pos[domain].extend(domain_recall_pos)
        all_domains_precision_pos[domain].extend(domain_precision_pos)

        all_domains_recall_neg[domain].extend(domain_recall_neg)
        all_domains_precision_neg[domain].extend(domain_precision_neg)

        all_domains_recall[domain].extend(domain_recall)
        all_domains_precision[domain].extend(domain_precision)

        all_domains_correctness[domain].extend(domain_correctness)
        all_domains_integrity[domain].extend(domain_integrity)


    # Create summary dataframe over all domains
    df = pd.DataFrame({'Domain': [],
                       'Instances': [],
                       'Avg time': [],
                       'Avg total objs': [],
                       'Avg ground actions': [],
                       'Avg executed actions': [],
                       'Avg success actions': [],
                       'Avg failed actions': [],
                       'Precs recall': [],
                       'Precs precision': [],
                       'Pos recall': [],
                       'Pos precision': [],
                       'Neg recall': [],
                       'Neg precision': [],
                       'Overall recall': [],
                       'Overall precision': [],
                       'Final correctness': [],
                       'Final integrity': []
                       })

    for domain in [el for el in all_domains if not el.startswith("overall")]:

        instances = len(all_domains_time[domain])
        avg_time = sum(all_domains_time[domain])/instances
        avg_total_objs = sum(all_domains_total_objs[domain])/instances
        avg_ground_actions = sum(all_domains_ground_actions[domain])/instances
        avg_executed_actions = sum(all_domains_executed_actions[domain])/instances
        avg_success_actions = sum(all_domains_success_actions[domain])/instances
        avg_failed_actions = sum(all_domains_failed_actions[domain])/instances

        final_recall_prec = all_domains_recall_prec[domain][-1]
        final_precision_prec = all_domains_precision_prec[domain][-1]

        final_recall_pos = all_domains_recall_pos[domain][-1]
        final_precision_pos = all_domains_precision_pos[domain][-1]

        final_recall_neg = all_domains_recall_neg[domain][-1]
        final_precision_neg = all_domains_precision_neg[domain][-1]

        final_recall = all_domains_recall[domain][-1]
        final_precision = all_domains_precision[domain][-1]

        final_correctness = all_domains_correctness[domain][-1]
        final_integrity = all_domains_integrity[domain][-1]



        evaluate = {'Domain': domain.split("_")[0],
                    'Instances': instances,
                    'Avg time': avg_time,
                    'Avg total objs': avg_total_objs,
                    'Avg ground actions': avg_ground_actions,
                    'Avg executed actions': avg_executed_actions,
                    'Avg success actions': avg_success_actions,
                    'Avg failed actions': avg_failed_actions,
                    'Precs recall': final_recall_prec,
                    'Precs precision': final_precision_prec,
                    'Pos recall': final_recall_pos,
                    'Pos precision': final_precision_pos,
                    'Neg recall': final_recall_neg,
                    'Neg precision': final_precision_neg,
                    'Overall recall': final_recall,
                    'Overall precision': final_precision,
                    'Final correctness': final_correctness,
                    'Final integrity': final_integrity
        }

        df = df.append(evaluate, ignore_index=True)


    # Create a Pandas Excel writer using XlsxWriter as the engine
    # writer = pd.ExcelWriter(os.path.join(RESULTS_DIR,"overall_summary.xlsx"), engine='xlsxwriter')
    # writer = pd.ExcelWriter(os.path.join(RESULTS_DIR,"overall_summary.xls"), engine='openpyxl')
    writer = pd.ExcelWriter(os.path.join(RESULTS_DIR,"overall_summary.xls"))
    # writer = pd.ExcelWriter(os.path.join(RESULTS_DIR,"overall_summary.xlsx"))
    # writer = pd.ExcelWriter(os.path.join(RESULTS_DIR,"overall_summary.xls"), engine='xlswriter')
    df.to_excel(writer, sheet_name='Summary', index=False, float_format = "%0.2f")
    writer.close()



def generate_comparison_latex_table(labels, file_name, caption, header):
    # folder = "Results"
    file_path = os.path.join("comparison_summary_uncertain.xlsx")

    with open(file_name + ".tex", "w") as f:
        df = pd.read_excel(file_path)
        df_restricted = df[labels]
        f.write(df_restricted.to_latex(index=False, escape=False,
                                       label="tab:{}".format(file_name),
                                       caption= caption,
                                       header = header))
                                       # .format(df_restricted.shape[0])))

        # f.write(df_restricted.to_latex(index=False))



def generate_comparison_latex_table_fama(labels, file_name, caption, header):
    # folder = "Results"
    file_path = os.path.join("comparison_fama.xlsx")

    with open(file_name + ".tex", "w") as f:
        df = pd.read_excel(file_path)
        df_restricted = df[labels]
        f.write(df_restricted.to_latex(index=False, escape=False,
                                       label="tab:{}".format(file_name),
                                       caption= caption,
                                       header = header))
                                       # .format(df_restricted.shape[0])))

        # f.write(df_restricted.to_latex(index=False))


def generate_summary_latex_table(labels, file_name, caption, header):
    folder = "Results"
    file_path = os.path.join(folder, "overall_summary.xlsx")

    with open(file_name + ".tex", "w") as f:
        df = pd.read_excel(file_path, sheet_name="Summary")
        df_restricted = df[labels]
        f.write(df_restricted.to_latex(index=False, escape=False,
                                       label="tab:{}".format(file_name),
                                       caption= caption,
                                       header = header))
                                       # .format(df_restricted.shape[0])))

        # f.write(df_restricted.to_latex(index=False))



if __name__ == "__main__":

# Generate domain dataframes and summary
    generate_domain_dataframes()

    generate_domain_summary()
#
#
#
# Generate summary (latex table)
    # labels = ["Domain", "Instances", "Precs precision",  "Precs recall","Pos precision", "Pos recall",
    #           "Neg precision", "Neg recall", "Overall precision", "Overall recall"]
    # header = ["Domain", "$I$", "$P_{\\prec}$", "$R_{\\prec}$", "$P_{\\eff^{+}}$", "$R_{\\eff^{+}}$", "$P_{\\eff^{-}}$",
    #           "$R_{\\eff^{-}}$", "$P$", "$R$"]
    # caption = "For each domain:statistics on final metrics of the last instance grouped by " \
    #           "preconditions, positive effects and negative ones."
    # file_name = "overall_summary_uncertain_nostripsass"
    # generate_summary_latex_table(labels, file_name, caption, header)
#
#
#
#
#
# Generate comparison summary uncertain negative effects (latex table)
    labels = ["Domain", "Neg precision A", "Neg recall A", "Overall precision A", "Overall recall A",
              "Neg precision B", "Neg recall B", "Overall precision B", "Overall recall B"]
    header = ["Domain", "$P_{\\eff^{-}}$", "$R_{\\eff^{-}}$", "$P$", "$R$",
              "$P_{\\eff^{-}}$", "$R_{\\eff^{-}}$", "$P$", "$R$"]
    caption = "For each domain:statistics on final metrics of the last instance grouped by " \
              "negative effects."
    file_name = "comparison_summary_uncertain"
    generate_comparison_latex_table(labels, file_name, caption, header)
#
#
# Generate FAMA average results
#     generate_fama_average_results()
#
# Generate comparison with FAMA (latex table)
#     labels = ["Domain", "Tot time", "Overall precision", "Overall recall", "FAMA tot time",
#               "FAMA precision", "FAMA recall", "Delta act"]
#     header = ["Domain", "$t$", "$P$", "$R$", "$t$", "$P$", "$R$", "$\delta_{A}$"]
#     caption = "Comparison among OLAM and FAMA with full observability. FAMA is run with all plan traces " \
#               "provided in \protect\cite{aineto_AIJ2019}. MODEL WITH UNCERTAIN NEGATIVE EFFECTS AND STRIPS ASSUMPTION."
#     file_name = "comparison_fama"
#     generate_comparison_latex_table_fama(labels, file_name, caption, header)
#
#
#
#
# Generate correctness and integrity (latex table)
    # labels = ["Domain", "Instances", "Avg time",  "Final correctness","Final integrity"]
    # header = ["Domain", "$I$", "$CPU_{time}$", "$C$", "$I$"]
    # caption = "For each domain: evaluation on average CPU time, correctness and integrity of the last instance."
    # file_name = "overall_corrint_certain"
    # generate_summary_latex_table(labels, file_name, caption, header)
