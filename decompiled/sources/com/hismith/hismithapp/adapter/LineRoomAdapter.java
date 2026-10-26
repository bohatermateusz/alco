package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.socket.SocketRoomModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class LineRoomAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    public static final int VIEW_TYPE_EMPTY = 0;
    public static final int VIEW_TYPE_ITEM = 1;
    private Context mContext;
    private List<SocketRoomModel> mRoomList;
    private OnItemClickListener onItemClickListener;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgRoomLock;
        private CircleImageView mImgUserHead;
        private LinearLayout mLayout;
        private TextView mTxtEnter;
        private TextView mTxtRoomId;
        private TextView mTxtRoomName;
        private TextView mTxtUserCount;
        private TextView mTxtUserName;

        public MyViewHolder(View view) {
            super(view);
            this.mLayout = (LinearLayout) view.findViewById(R.id.item_line_room_layout);
            this.mImgUserHead = (CircleImageView) view.findViewById(R.id.item_line_room_img_user_head);
            this.mImgRoomLock = (ImageView) view.findViewById(R.id.item_line_room_img_room_lock);
            this.mTxtUserName = (TextView) view.findViewById(R.id.item_line_room_txt_user_name);
            this.mTxtRoomName = (TextView) view.findViewById(R.id.item_line_room_txt_room_name);
            this.mTxtRoomId = (TextView) view.findViewById(R.id.item_line_room_txt_room_id);
            this.mTxtUserCount = (TextView) view.findViewById(R.id.item_line_room_txt_user_count);
            this.mTxtEnter = (TextView) view.findViewById(R.id.item_line_room_txt_enter);
            LanguageUtil.setText(LineRoomAdapter.this.mContext, this.mTxtEnter, "joinButton");
        }
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    public LineRoomAdapter(Context context, List<SocketRoomModel> list) {
        this.mContext = context;
        this.mRoomList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mRoomList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        return 1;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(RecyclerView.ViewHolder viewHolder, final int i) {
        if (viewHolder instanceof MyViewHolder) {
            final MyViewHolder myViewHolder = (MyViewHolder) viewHolder;
            Glide.with(this.mContext).load(this.mRoomList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.LineRoomAdapter.2
                public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                    myViewHolder.mImgUserHead.setImageDrawable(drawable);
                }

                @Override // com.bumptech.glide.request.target.Target
                public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                    onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                }
            });
            myViewHolder.mTxtUserName.setText(this.mRoomList.get(i).getUserName());
            myViewHolder.mTxtRoomName.setText(this.mRoomList.get(i).getRoomName());
            myViewHolder.mTxtRoomId.setText(this.mRoomList.get(i).getRoomId() + "");
            myViewHolder.mTxtUserCount.setText(this.mRoomList.get(i).getUserCount() + "-" + this.mRoomList.get(i).getUserLimit());
            if (this.mRoomList.get(i).getIsPublic() == 1) {
                myViewHolder.mImgRoomLock.setVisibility(8);
            } else {
                myViewHolder.mImgRoomLock.setVisibility(0);
            }
            if (this.mRoomList.get(i).getRoomPlay() == 0) {
                myViewHolder.mTxtEnter.setTextColor(this.mContext.getResources().getColor(R.color.white40));
                myViewHolder.mTxtEnter.setBackgroundResource(R.drawable.btn_join_room_keepout);
                myViewHolder.mTxtEnter.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.LineRoomAdapter.3
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        ToastUtil.showShortToast(LineRoomAdapter.this.mContext, LanguageUtil.getLanguageByKey(LineRoomAdapter.this.mContext, "roomIncompatibility"));
                    }
                });
            } else {
                myViewHolder.mTxtEnter.setTextColor(this.mContext.getResources().getColor(R.color.white));
                myViewHolder.mTxtEnter.setBackgroundResource(R.drawable.btn_join_room);
                myViewHolder.mTxtEnter.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.LineRoomAdapter.4
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        if (LineRoomAdapter.this.onItemClickListener != null) {
                            LineRoomAdapter.this.onItemClickListener.onItemClick(i);
                        }
                    }
                });
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return i == 0 ? new RecyclerView.ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.empty_layout, viewGroup, false)) { // from class: com.hismith.hismithapp.adapter.LineRoomAdapter.1
        } : new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_line_room, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
