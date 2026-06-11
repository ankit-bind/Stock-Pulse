import os

with open('d:/Stock-Pulse/dashboard/views/ml_strategy.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update Feature Importances
old_imp = '''    if importance_mean:
        st.subheader("Feature importances (mean across walk-forward fits)")
        for mname, ser in importance_mean.items():
            st.markdown(f"**{mname}**")
            st.bar_chart(ser)
            if mname in importance_std and importance_std[mname].notna().any():
                with st.expander(f"{mname} — importance std across folds"):
                    st.bar_chart(importance_std[mname].fillna(0.0))'''

new_imp = '''    if importance_mean:
        st.subheader("Feature importances (mean across walk-forward fits)")
        for mname, ser in importance_mean.items():
            df_imp = ser.reset_index()
            df_imp.columns = ['Feature', 'Importance']
            df_imp = df_imp.sort_values(by='Importance', ascending=True)
            fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title=mname, color_discrete_sequence=['#00d4ff'])
            fig_imp.update_layout(height=max(400, len(df_imp) * 25), showlegend=False, xaxis_title="Mean Importance", yaxis_title="")
            st.plotly_chart(apply_theme(fig_imp), use_container_width=True)
            
            if mname in importance_std and importance_std[mname].notna().any():
                with st.expander(f"{mname} — importance std across folds"):
                    df_std = importance_std[mname].fillna(0.0).reset_index()
                    df_std.columns = ['Feature', 'StdDev']
                    df_std = df_std.sort_values(by='StdDev', ascending=True)
                    fig_std = px.bar(df_std, x='StdDev', y='Feature', orientation='h', color_discrete_sequence=['#7c3aed'])
                    fig_std.update_layout(height=max(400, len(df_std) * 25), showlegend=False, xaxis_title="Std Deviation", yaxis_title="")
                    st.plotly_chart(apply_theme(fig_std), use_container_width=True)'''

if old_imp in content:
    content = content.replace(old_imp, new_imp)
    print("Updated Feature Importances")
else:
    print("Could not find feature importances block")

# Update Strategy Comparison Bubble Chart
# We insert it right before the disp.style.format (which we already replaced)
old_comp = 'st.subheader("Decision layer — strategy comparison (OOS window)")\n    comp = pd.DataFrame(rows)'
new_comp = '''st.subheader("Decision layer — strategy comparison (OOS window)")
    comp = pd.DataFrame(rows)
    if not comp.empty:
        # Create Bubble Chart
        bubble_df = comp.copy().dropna(subset=["CAGR", "Max DD", "Sharpe (ann.)"])
        if not bubble_df.empty:
            # We must make size strictly positive for plotly scatter
            bubble_df["BubbleSize"] = bubble_df["Sharpe (ann.)"].clip(lower=0) + 0.1 
            fig_bubble = px.scatter(
                bubble_df, x="Max DD", y="CAGR", size="BubbleSize", color="CAGR",
                hover_name="Strategy", color_continuous_scale="RdYlGn",
                title="Risk vs Return Profile (Size = Sharpe)",
                labels={"Max DD": "Max Drawdown", "CAGR": "CAGR"}
            )
            fig_bubble.update_layout(height=500)
            st.plotly_chart(apply_theme(fig_bubble), use_container_width=True)'''

if old_comp in content:
    content = content.replace(old_comp, new_comp)
    print("Added Strategy Comparison Bubble Chart")
else:
    print("Could not find Strategy Comparison block")

with open('d:/Stock-Pulse/dashboard/views/ml_strategy.py', 'w', encoding='utf-8') as f:
    f.write(content)

