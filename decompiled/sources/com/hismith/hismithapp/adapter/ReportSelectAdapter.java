package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import java.util.List;

/* loaded from: classes3.dex */
public class ReportSelectAdapter extends RecyclerView.Adapter<ViewHolder> {
    private int clickPosition = -1;
    private Context mContext;
    private List<String> mList;

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgButton;
        private TextView mTxtItem;

        public ViewHolder(View view) {
            super(view);
            this.mImgButton = (ImageView) view.findViewById(R.id.img_button);
            this.mTxtItem = (TextView) view.findViewById(R.id.txt_item);
        }
    }

    public ReportSelectAdapter(Context context, List<String> list) {
        this.mContext = context;
        this.mList = list;
    }

    public int getClickPosition() {
        return this.clickPosition;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.mTxtItem.setText(this.mList.get(i));
        if (this.clickPosition == i) {
            viewHolder.mImgButton.setImageResource(R.mipmap.radio_checked);
        } else {
            viewHolder.mImgButton.setImageResource(R.mipmap.radio_normal);
        }
        viewHolder.mImgButton.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ReportSelectAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                ReportSelectAdapter.this.clickPosition = i;
                ReportSelectAdapter.this.notifyDataSetChanged();
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_report_select, viewGroup, false));
    }
}
