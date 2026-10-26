package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.ChatUpAdapter;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.interfaces.OnSelectListener;
import java.util.List;

/* loaded from: classes3.dex */
public class MessageAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private Context mContext;
    private List<MessageModel.Message> mList;
    private OnDeleteClickListener onDeleteClickListener;
    private ChatUpAdapter.OnItemCLickListener onItemCLickListener;
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

    public interface OnDeleteClickListener {
        void onDeleteClick(int i);
    }

    public interface OnItemCLickListener {
        void onItemClick(int i);
    }

    public MessageAdapter(Context context, List<MessageModel.Message> list) {
        this.mContext = context;
        this.mList = list;
    }

    private View getEmptyView(ViewGroup viewGroup) {
        LogUtil.e("空数据getEmptyView");
        return LayoutInflater.from(this.mContext).inflate(R.layout.layout_empty, viewGroup, false);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<MessageModel.Message> list = this.mList;
        int size = list != null ? list.size() : 0;
        return size > 0 ? size : this.showEmptyView ? 1 : 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        return isEmptyPosition(i) ? 1 : 2;
    }

    public boolean isEmptyPosition(int i) {
        List<MessageModel.Message> list = this.mList;
        return i == 0 && this.showEmptyView && (list != null ? list.size() : 0) == 0;
    }

    public boolean isShowEmptyView() {
        return this.showEmptyView;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final MyViewHolder myViewHolder, final int i) {
        if (isEmptyPosition(i)) {
            LogUtil.e("空数据onBindViewHolder" + i);
            if (myViewHolder.mEmptyTxt1 != null) {
                LanguageUtil.setText(this.mContext, myViewHolder.mEmptyTxt1, "noChatData");
                return;
            }
            return;
        }
        myViewHolder.mImgHead.setImageResource(R.mipmap.ic_personal_head);
        Glide.with(this.mContext).load(this.mList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.MessageAdapter.1
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                myViewHolder.mImgHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        myViewHolder.mImgHead.setKing(this.mList.get(i).getKing(), this.mContext, this.mList.get(i).getGradeIcon());
        LogUtil.e("messageAdapter " + i + " alias:" + this.mList.get(i).getAlias() + ", name:" + this.mList.get(i).getName());
        myViewHolder.mTxtName.setText(this.mList.get(i).getAlias());
        myViewHolder.mTxtTime.setText(TimeUtil.utc2Local(this.mList.get(i).getDate()));
        if (this.mList.get(i).getAndroidRead() == 0) {
            myViewHolder.mTxtCount.setVisibility(0);
            myViewHolder.mTxtCount.setText(this.mList.get(i).getAndroidReadCount() + "");
        } else {
            myViewHolder.mTxtCount.setVisibility(8);
        }
        if ("text".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText(this.mList.get(i).getContent());
        } else if ("image".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "image") + "]");
        } else if ("voice".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "voice") + "]");
        } else if (Constans.MESSAGE_TYPE_REMOTE.equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "remotePlay") + "]");
        } else if ("video".equals(this.mList.get(i).getType())) {
            myViewHolder.mTxtContent.setText("[" + LanguageUtil.getLanguageByKey(this.mContext, "video") + "]");
        }
        myViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MessageAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (MessageAdapter.this.onItemCLickListener != null) {
                    myViewHolder.mTxtCount.setVisibility(8);
                    ((MessageModel.Message) MessageAdapter.this.mList.get(i)).setAndroidRead(1);
                    ((MessageModel.Message) MessageAdapter.this.mList.get(i)).setAndroidReadCount(0);
                    MessageAdapter.this.onItemCLickListener.onItemClick(i);
                }
            }
        });
        final XPopup.Builder watchView = new XPopup.Builder(this.mContext).watchView(myViewHolder.itemView);
        myViewHolder.itemView.setOnLongClickListener(new View.OnLongClickListener() { // from class: com.hismith.hismithapp.adapter.MessageAdapter.3
            @Override // android.view.View.OnLongClickListener
            public boolean onLongClick(View view) {
                watchView.asAttachList(new String[]{LanguageUtil.getLanguageByKey(MessageAdapter.this.mContext, "delete")}, null, new OnSelectListener() { // from class: com.hismith.hismithapp.adapter.MessageAdapter.3.1
                    @Override // com.lxj.xpopup.interfaces.OnSelectListener
                    public void onSelect(int i2, String str) {
                        if (i2 != 0 || MessageAdapter.this.onDeleteClickListener == null) {
                            return;
                        }
                        MessageAdapter.this.onDeleteClickListener.onDeleteClick(i);
                    }
                }).show();
                return true;
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return i == 1 ? new MyViewHolder(getEmptyView(viewGroup)) : new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_message, viewGroup, false));
    }

    public void setOnDeleteClickListener(OnDeleteClickListener onDeleteClickListener) {
        this.onDeleteClickListener = onDeleteClickListener;
    }

    public void setOnItemCLickListener(ChatUpAdapter.OnItemCLickListener onItemCLickListener) {
        this.onItemCLickListener = onItemCLickListener;
    }

    public void setShowEmptyView(boolean z) {
        this.showEmptyView = z;
    }
}
