package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.FirstChatModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class ChatUpAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private Context mContext;
    private List<FirstChatModel> mList;
    private OnItemCLickListener onItemCLickListener;
    private final int VIEW_TYPE_EMPTY = 1;
    private final int VIEW_TYPE_DATA = 2;
    private boolean showEmptyView = false;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private TextView mEmptyTxt1;
        private CircleImageView mImgHead;
        private TextView mTxtContent;
        private TextView mTxtCount;
        private TextView mTxtName;
        private TextView mTxtTime;

        public MyViewHolder(View view) {
            super(view);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.message_img_head);
            this.mTxtName = (TextView) view.findViewById(R.id.message_txt_name);
            this.mTxtTime = (TextView) view.findViewById(R.id.message_txt_time);
            this.mTxtContent = (TextView) view.findViewById(R.id.message_txt_content);
            this.mTxtCount = (TextView) view.findViewById(R.id.message_txt_count);
            this.mEmptyTxt1 = (TextView) view.findViewById(R.id.empty_txt1);
        }
    }

    public interface OnItemCLickListener {
        void onItemClick(int i);
    }

    public ChatUpAdapter(Context context, List<FirstChatModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    private View getEmptyView(ViewGroup viewGroup) {
        LogUtil.e("空数据getEmptyView");
        return LayoutInflater.from(this.mContext).inflate(R.layout.layout_empty, viewGroup, false);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<FirstChatModel> list = this.mList;
        int size = list != null ? list.size() : 0;
        return size > 0 ? size : this.showEmptyView ? 1 : 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        return isEmptyPosition(i) ? 1 : 2;
    }

    public boolean isEmptyPosition(int i) {
        List<FirstChatModel> list = this.mList;
        return i == 0 && this.showEmptyView && (list != null ? list.size() : 0) == 0;
    }

    public boolean isShowEmptyView() {
        return this.showEmptyView;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(MyViewHolder myViewHolder, final int i) {
        if (isEmptyPosition(i)) {
            LogUtil.e("空数据onBindViewHolder" + i);
            if (myViewHolder.mEmptyTxt1 != null) {
                LanguageUtil.setText(this.mContext, myViewHolder.mEmptyTxt1, "noChatData");
                return;
            }
            return;
        }
        Glide.with(this.mContext).load(this.mList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(myViewHolder.mImgHead);
        myViewHolder.mImgHead.setKing(this.mList.get(i).getKing(), this.mContext, this.mList.get(i).getGradeIcon());
        myViewHolder.mTxtName.setText(this.mList.get(i).getName());
        myViewHolder.mTxtTime.setText(TimeUtil.utc2Local(this.mList.get(i).getDate()));
        if ("text".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText(this.mList.get(i).getContent());
        } else if ("image".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "image") + "]");
        } else if ("voice".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "voice") + "]");
        } else if (Constans.MESSAGE_TYPE_REMOTE.equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "remotePlay") + "]");
        }
        myViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatUpAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ChatUpAdapter.this.onItemCLickListener != null) {
                    ChatUpAdapter.this.onItemCLickListener.onItemClick(i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return i == 1 ? new MyViewHolder(getEmptyView(viewGroup)) : new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_message, viewGroup, false));
    }

    public void setOnItemCLickListener(OnItemCLickListener onItemCLickListener) {
        this.onItemCLickListener = onItemCLickListener;
    }

    public void setShowEmptyView(boolean z) {
        this.showEmptyView = z;
    }
}
