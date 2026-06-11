import os
print(os.getcwd())

with open('d:/Stock-Pulse/dashboard/views/ml_strategy.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''        st.dataframe(
            disp.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Sharpe (ann.)": "{:.2f}",
                    "Max DD": "{:.2%}",
                    "Turnover": "{:.3f}",
                    "IC (Spearman, OOS)": "{:.3f}",
                    "RMSE": "{:.6f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
        )'''

new_block = '''        styled = (disp.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Sharpe (ann.)": "{:.2f}",
                    "Max DD": "{:.2%}",
                    "Turnover": "{:.3f}",
                    "IC (Spearman, OOS)": "{:.3f}",
                    "RMSE": "{:.6f}",
                },
                na_rep="—",
            )
            .applymap(lambda v: "color: #10b981" if isinstance(v, (int, float)) and v > 0 else ("color: #ef4444" if isinstance(v, (int, float)) and v < 0 else ""), subset=["CAGR", "Sharpe (ann.)", "IC (Spearman, OOS)"])
            .set_properties(**{"font-family": "JetBrains Mono"})
        )
        st.dataframe(styled, use_container_width=True)
        st.download_button(label="Export CSV", data=disp.to_csv().encode('utf-8'), file_name='strategy_comparison.csv', mime='text/csv', key='comp_dl')'''

if old_block in content:
    content = content.replace(old_block, new_block)
    print("Updated ML strategy comparison table.")
else:
    print("Could not find ML strategy comparison table.")


old_block_2 = "st.dataframe(panel.tail(8), use_container_width=True)"
new_block_2 = '''styled_panel = (panel.tail(8).style
                        .applymap(lambda v: "color: #10b981" if isinstance(v, (int, float)) and v > 0 else ("color: #ef4444" if isinstance(v, (int, float)) and v < 0 else ""), subset=[c for c in panel.columns if "pred" in c or "ret" in c])
                        .set_properties(**{"font-family": "JetBrains Mono"}))
                    st.dataframe(styled_panel, use_container_width=True)
                    st.download_button(label="Export Panel CSV", data=panel.to_csv(index=False).encode('utf-8'), file_name='panel_results.csv', mime='text/csv', key='panel_dl')'''

if old_block_2 in content:
    content = content.replace(old_block_2, new_block_2)
    print("Updated ML panel table.")

with open('d:/Stock-Pulse/dashboard/views/ml_strategy.py', 'w', encoding='utf-8') as f:
    f.write(content)


with open('d:/Stock-Pulse/dashboard/views/stock_analysis.py', 'r', encoding='utf-8') as f:
    content_sa = f.read()

old_block_sa = 'st.dataframe(comp_df.style.format({"Strategy CAGR": "{:.2%}", "Total Return": "{:.2%}"}), use_container_width=True)'
new_block_sa = '''styled_comp = (comp_df.style.format({"Strategy CAGR": "{:.2%}", "Total Return": "{:.2%}"})
                     .applymap(lambda v: "color: #10b981" if isinstance(v, (int, float)) and v > 0 else ("color: #ef4444" if isinstance(v, (int, float)) and v < 0 else ""), subset=["Strategy CAGR", "Total Return"])
                     .set_properties(**{"font-family": "JetBrains Mono"}))
                st.dataframe(styled_comp, use_container_width=True)
                st.download_button(label="Download Comparison CSV", data=comp_df.to_csv(index=False).encode('utf-8'), file_name='comparison.csv', mime='text/csv', key='stock_comp_dl')'''

if old_block_sa in content_sa:
    content_sa = content_sa.replace(old_block_sa, new_block_sa)
    print("Updated Stock Analysis comparison table.")

with open('d:/Stock-Pulse/dashboard/views/stock_analysis.py', 'w', encoding='utf-8') as f:
    f.write(content_sa)

