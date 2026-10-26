package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.listener.CustomClickListener;
import com.hismith.hismithapp.model.LanguageModel;
import java.util.List;

/* loaded from: classes3.dex */
public class LanguageAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private Context mContext;
    private List<LanguageModel> mList;
    private OnItemClickListener onItemClickListener;
    private Integer textColor;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private View mItemLine;
        private TextView mItemTxtLanguage;

        public MyViewHolder(View view) {
            super(view);
            this.mItemTxtLanguage = (TextView) view.findViewById(R.id.item_txt_language);
            this.mItemLine = view.findViewById(R.id.item_line);
        }
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    public LanguageAdapter(Context context, List<LanguageModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    public LanguageAdapter(Context context, List<LanguageModel> list, int i) {
        this.mContext = context;
        this.mList = list;
        this.textColor = Integer.valueOf(i);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(MyViewHolder myViewHolder, final int i) {
        if (this.textColor != null) {
            myViewHolder.mItemTxtLanguage.setTextColor(this.textColor.intValue());
        }
        myViewHolder.mItemTxtLanguage.setText(this.mList.get(i).getName());
        myViewHolder.itemView.setOnClickListener(new CustomClickListener() { // from class: com.hismith.hismithapp.adapter.LanguageAdapter.1
            @Override // com.hismith.hismithapp.listener.CustomClickListener
            public void onSingleClick(View view) {
                if (LanguageAdapter.this.onItemClickListener != null) {
                    LanguageAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
        if (i == this.mList.size() - 1) {
            myViewHolder.mItemLine.setVisibility(8);
        } else {
            myViewHolder.mItemLine.setVisibility(0);
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_login_language, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
