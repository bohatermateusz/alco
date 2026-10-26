package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.ModeBlockModel;
import com.hismith.hismithapp.surface_view.LineDataUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;
import com.hismith.hismithapp.view.LinePointsView;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayModeListAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<ModeBlockModel.ModeBlock> mList;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private LinePointsView line_points_view;
        private TextView txt_mode_name;
        private TextView txt_mode_time;

        public ViewHolder(View view) {
            super(view);
            this.txt_mode_name = (TextView) view.findViewById(R.id.txt_mode_name);
            this.line_points_view = (LinePointsView) view.findViewById(R.id.line_points_view);
            this.txt_mode_time = (TextView) view.findViewById(R.id.txt_mode_time);
            TextView textView = this.txt_mode_name;
            if (textView != null) {
                textView.setVisibility(8);
            }
        }
    }

    public PlayModeListAdapter(Context context, List<ModeBlockModel.ModeBlock> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        String[] buildServerToClientData = LineDataUtil.buildServerToClientData(this.mList.get(i).getBlockData(), this.mList.get(i).getBlockId());
        viewHolder.line_points_view.setStretch(true);
        if (buildServerToClientData == null || TextUtils.isEmpty(buildServerToClientData[0])) {
            viewHolder.line_points_view.setData("", "", false);
        } else {
            viewHolder.line_points_view.setData(buildServerToClientData, false);
        }
        viewHolder.txt_mode_time.setText(ModeTimeUtil.formatMilliTime((int) (this.mList.get(i).getBlockTime() * 1000.0d)));
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeListAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (PlayModeListAdapter.this.onItemClickListener != null) {
                    PlayModeListAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.adapter_mode_list, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
