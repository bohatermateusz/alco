package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.AnimationDrawable;
import android.graphics.drawable.Drawable;
import android.media.MediaPlayer;
import android.text.SpannableStringBuilder;
import android.text.TextPaint;
import android.text.TextUtils;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.load.resource.bitmap.RoundedCorners;
import com.bumptech.glide.request.BaseRequestOptions;
import com.bumptech.glide.request.RequestOptions;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.facebook.internal.AnalyticsEvents;
import com.hismith.hismithapp.ChatActivity;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.image_preview.glide.progress.OnProgressListener;
import com.hismith.hismithapp.image_preview.glide.progress.ProgressManager;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.ReferenceMsg;
import com.hismith.hismithapp.popup.ChatAttachPopup;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.luck.picture.lib.config.PictureMimeType;
import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import jp.wasabeef.glide.transformations.RoundedCornersTransformation;

/* loaded from: classes3.dex */
public class ChatAdapter extends RecyclerView.Adapter<ViewHolder> implements MediaPlayer.OnCompletionListener, MediaPlayer.OnPreparedListener {
    private ClickCallback clickCallback;
    private ChatAttachPopup mChatAttachPopup;
    private List<MessageModel.Message> mChatList;
    private Context mContext;
    private LinearLayoutManager mLayoutManager;
    private MediaPlayer mediaPlayer;
    private OnActionClickListener onActionClickListener;
    private int playPosition = -1;
    private boolean isFirstLoad = true;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    public interface ClickCallback {
        void onClickUser(int i);
    }

    class ImagePreviewClickListener implements View.OnClickListener {
        String filePath;

        public ImagePreviewClickListener(String str) {
            this.filePath = str;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            ImagePreview.getInstance().setContext(ChatAdapter.this.mContext).setImage(this.filePath).setFolderName("hismith/image").setEnableDragClose(true).start();
        }
    }

    public interface OnActionClickListener {
        void onCopy(int i);

        void onDelete(int i);

        void onReference(int i);

        void onRevoke(int i);
    }

    class OnRevokeClickListener implements View.OnLongClickListener {
        private int mPosition;

        public OnRevokeClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnLongClickListener
        public boolean onLongClick(View view) {
            LogUtil.e("OnRevokeClickListener");
            boolean equals = "image".equals(((MessageModel.Message) ChatAdapter.this.mChatList.get(this.mPosition)).getType());
            ChatAdapter chatAdapter = ChatAdapter.this;
            if (equals) {
                chatAdapter.mChatAttachPopup.setShowCopy(false);
            } else {
                boolean equals2 = "voice".equals(((MessageModel.Message) chatAdapter.mChatList.get(this.mPosition)).getType());
                ChatAdapter chatAdapter2 = ChatAdapter.this;
                if (equals2) {
                    chatAdapter2.mChatAttachPopup.setShowCopy(false);
                } else {
                    chatAdapter2.mChatAttachPopup.setShowCopy(true);
                }
            }
            ChatAdapter.this.mChatAttachPopup.setShowRevoke(false);
            if (this.mPosition >= 0) {
                try {
                    Date parse = ChatAdapter.this.sdf.parse(((MessageModel.Message) ChatAdapter.this.mChatList.get(this.mPosition)).getDate());
                    if (parse != null) {
                        LogUtil.e("撤回时间：" + parse.toString() + ", 和现在相差" + (System.currentTimeMillis() - parse.getTime()));
                        if (System.currentTimeMillis() - parse.getTime() < 120000 && ((MessageModel.Message) ChatAdapter.this.mChatList.get(this.mPosition)).getAndroidSend() == 1) {
                            ChatAdapter.this.mChatAttachPopup.setShowRevoke(true);
                        }
                    }
                } catch (ParseException e) {
                    e.printStackTrace();
                }
            }
            ChatAdapter.this.mChatAttachPopup.setOnActionClickListener(new ChatAttachPopup.OnActionClickListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.OnRevokeClickListener.1
                @Override // com.hismith.hismithapp.popup.ChatAttachPopup.OnActionClickListener
                public void onCopy(int i) {
                    if (ChatAdapter.this.onActionClickListener != null) {
                        ChatAdapter.this.onActionClickListener.onCopy(OnRevokeClickListener.this.mPosition);
                    }
                    LogUtil.e("复制：" + ((MessageModel.Message) ChatAdapter.this.mChatList.get(OnRevokeClickListener.this.mPosition)).getMsgId());
                }

                @Override // com.hismith.hismithapp.popup.ChatAttachPopup.OnActionClickListener
                public void onDelete(int i) {
                    if (ChatAdapter.this.onActionClickListener != null) {
                        ChatAdapter.this.onActionClickListener.onDelete(OnRevokeClickListener.this.mPosition);
                    }
                    LogUtil.e("删除：" + ((MessageModel.Message) ChatAdapter.this.mChatList.get(OnRevokeClickListener.this.mPosition)).getMsgId());
                }

                @Override // com.hismith.hismithapp.popup.ChatAttachPopup.OnActionClickListener
                public void onReference(int i) {
                    if (ChatAdapter.this.onActionClickListener != null) {
                        ChatAdapter.this.onActionClickListener.onReference(OnRevokeClickListener.this.mPosition);
                    }
                    LogUtil.e("引用：" + ((MessageModel.Message) ChatAdapter.this.mChatList.get(OnRevokeClickListener.this.mPosition)).getMsgId());
                }

                @Override // com.hismith.hismithapp.popup.ChatAttachPopup.OnActionClickListener
                public void onRevoke(int i) {
                    if (ChatAdapter.this.onActionClickListener != null) {
                        ChatAdapter.this.onActionClickListener.onRevoke(OnRevokeClickListener.this.mPosition);
                    }
                    LogUtil.e("撤回：" + ((MessageModel.Message) ChatAdapter.this.mChatList.get(OnRevokeClickListener.this.mPosition)).getMsgId());
                }
            });
            ChatAdapter.this.mChatAttachPopup.showPop(this.mPosition, view, MeasureUtil.dip2px(ChatAdapter.this.mContext, 0.0f));
            return true;
        }
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ViewGroup chat_layout_reference;
        private ImageView item_reference_image;
        private ImageView item_reference_img_play;
        private ConstraintLayout item_reference_layout_voice;
        private TextView item_reference_text;
        private TextView item_reference_txt_time;
        private ImageView mItemImgContent;
        private ImageView mItemImgError;
        private CircleImageView mItemImgHead;
        private ImageView mItemImgPlay;
        private FrameLayout mItemLayoutVoice;
        private ProgressBar mItemProgress;
        private TextView mItemTxtContent;
        private TextView mItemTxtRevoke;
        private TextView mItemTxtSendTime;
        private TextView mItemTxtTime;

        public ViewHolder(View view) {
            super(view);
            this.mItemImgHead = (CircleImageView) view.findViewById(R.id.chat_item_img_head);
            this.mItemTxtContent = (TextView) view.findViewById(R.id.chat_item_txt_content);
            this.mItemImgContent = (ImageView) view.findViewById(R.id.chat_item_img_content);
            this.mItemProgress = (ProgressBar) view.findViewById(R.id.chat_item_progress);
            this.mItemImgError = (ImageView) view.findViewById(R.id.chat_item_img_error);
            this.mItemLayoutVoice = (FrameLayout) view.findViewById(R.id.chat_item_layout_voice);
            this.mItemTxtTime = (TextView) view.findViewById(R.id.chat_item_txt_time);
            this.mItemImgPlay = (ImageView) view.findViewById(R.id.chat_item_img_play);
            this.mItemTxtSendTime = (TextView) view.findViewById(R.id.chat_item_txt_send_time);
            this.mItemTxtRevoke = (TextView) view.findViewById(R.id.txt_revoke);
            this.chat_layout_reference = (ViewGroup) view.findViewById(R.id.chat_layout_reference);
            this.item_reference_text = (TextView) view.findViewById(R.id.item_reference_text);
            this.item_reference_image = (ImageView) view.findViewById(R.id.item_reference_image);
            this.item_reference_layout_voice = (ConstraintLayout) view.findViewById(R.id.item_reference_layout_voice);
            this.item_reference_img_play = (ImageView) view.findViewById(R.id.item_reference_img_play);
            this.item_reference_txt_time = (TextView) view.findViewById(R.id.item_reference_txt_time);
        }
    }

    class VoicePlayClickListener implements View.OnClickListener {
        private int playMsgId;
        private String referenceContent;

        public VoicePlayClickListener(int i, String str) {
            this.playMsgId = i;
            this.referenceContent = str;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            LogUtil.e("VoicePlayClickListener msgId:" + this.playMsgId);
            boolean z = false;
            for (int i = 0; i < ChatAdapter.this.mChatList.size(); i++) {
                LogUtil.e("VoicePlayClickListener if(" + i + "):" + ((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getMsgId() + "==" + this.playMsgId);
                if (((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getMsgId() == this.playMsgId) {
                    LogUtil.e("VoicePlayClickListener playPosition:" + ChatAdapter.this.playPosition);
                    if (ChatAdapter.this.playPosition == i) {
                        LogUtil.e("VoicePlayClickListener releasePlay");
                        ChatAdapter.this.playPosition = -1;
                        ChatAdapter.this.releasePlay();
                    } else {
                        LogUtil.e("VoicePlayClickListener playVoice:" + ((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getContent());
                        ChatAdapter.this.playPosition = i;
                        ChatAdapter chatAdapter = ChatAdapter.this;
                        chatAdapter.playVoice(((MessageModel.Message) chatAdapter.mChatList.get(i)).getContent());
                    }
                    ChatAdapter.this.notifyItemChanged(i);
                    z = true;
                }
            }
            if (z) {
                return;
            }
            ChatAdapter.this.playVoice(this.referenceContent);
        }
    }

    public ChatAdapter(Context context, LinearLayoutManager linearLayoutManager, List<MessageModel.Message> list) {
        this.mContext = context;
        this.mLayoutManager = linearLayoutManager;
        this.mChatList = list;
        this.mChatAttachPopup = new ChatAttachPopup(this.mContext);
        this.sdf.setTimeZone(TimeZone.getTimeZone("GMT+00:00"));
    }

    private void callService(final String str, TextView textView) {
        SpannableStringBuilder spannableStringBuilder = new SpannableStringBuilder(str);
        spannableStringBuilder.setSpan(new ClickableSpan() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.8
            @Override // android.text.style.ClickableSpan
            public void onClick(View view) {
                if (view.getId() == R.id.chat_item_txt_content) {
                    AppUtil.startUriToWeb(ChatAdapter.this.mContext, str);
                }
            }

            @Override // android.text.style.ClickableSpan, android.text.style.CharacterStyle
            public void updateDrawState(TextPaint textPaint) {
                super.updateDrawState(textPaint);
                textPaint.setColor(Color.parseColor("#006EF5"));
                textPaint.setUnderlineText(true);
            }
        }, 0, str.length(), 33);
        textView.setHighlightColor(0);
        textView.setText(spannableStringBuilder);
        textView.setMovementMethod(LinkMovementMethod.getInstance());
    }

    /* JADX INFO: Access modifiers changed from: private */
    public int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, this.mContext.getResources().getDisplayMetrics());
    }

    private String getCompleteUrl(String str) {
        Matcher matcher = Pattern.compile("((http|ftp|https)://)(([a-zA-Z0-9\\._-]+\\.[a-zA-Z]{2,6})|([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\\&%_\\./-~-]*)?", 2).matcher(str);
        matcher.find();
        return matcher.group();
    }

    private boolean isHttpUrl(String str) {
        if (TextUtils.isEmpty(str)) {
            return false;
        }
        boolean matches = Pattern.compile("(((https|http)?://)?([a-z0-9]+[.])|(www.))\\w+[.|\\/]([a-z0-9]{0,})?[[.]([a-z0-9]{0,})]+((/[\\S&&[^,;一-龥]]+)+)?([.][a-z0-9]{0,}+|/?)".trim()).matcher(str.trim()).matches();
        if (matches) {
            return true;
        }
        return matches;
    }

    private void notifyItemReject(ViewHolder viewHolder, int i) {
        int reject = this.mChatList.get(i).getReject();
        List<MessageModel.Message> list = this.mChatList;
        if (reject == 3) {
            if (list.get(i).getMsgId() == SocketManager.getInstance().getMsgId()) {
                LanguageUtil.setText(this.mContext, viewHolder.mItemTxtContent, "playing");
                return;
            }
            int voiceLength = this.mChatList.get(i).getVoiceLength();
            Context context = this.mContext;
            if (voiceLength == 0) {
                LanguageUtil.setText(context, viewHolder.mItemTxtContent, "interruptedPlay");
                return;
            } else {
                viewHolder.mItemTxtContent.setText(LanguageUtil.getLanguageByKey(context, "playTime") + " " + ModeTimeUtil.formatMilliTime(this.mChatList.get(i).getVoiceLength()));
                return;
            }
        }
        int reject2 = list.get(i).getReject();
        List<MessageModel.Message> list2 = this.mChatList;
        if (reject2 == 2) {
            int androidSend = list2.get(i).getAndroidSend();
            Context context2 = this.mContext;
            if (androidSend == 1) {
                LanguageUtil.setText(context2, viewHolder.mItemTxtContent, AnalyticsEvents.PARAMETER_SHARE_OUTCOME_CANCELLED);
                return;
            } else {
                LanguageUtil.setText(context2, viewHolder.mItemTxtContent, "otherClosed");
                return;
            }
        }
        int reject3 = list2.get(i).getReject();
        List<MessageModel.Message> list3 = this.mChatList;
        if (reject3 == 1) {
            int androidSend2 = list3.get(i).getAndroidSend();
            Context context3 = this.mContext;
            if (androidSend2 == 1) {
                LanguageUtil.setText(context3, viewHolder.mItemTxtContent, "otherDeclined");
                return;
            } else {
                LanguageUtil.setText(context3, viewHolder.mItemTxtContent, "rejected");
                return;
            }
        }
        int androidSend3 = list3.get(i).getAndroidSend();
        Context context4 = this.mContext;
        if (androidSend3 == 1) {
            LanguageUtil.setText(context4, viewHolder.mItemTxtContent, "otherNoAnswer");
        } else {
            LanguageUtil.setText(context4, viewHolder.mItemTxtContent, AnalyticsEvents.PARAMETER_SHARE_OUTCOME_CANCELLED);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void playVoice(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        releasePlay();
        try {
            MediaPlayer mediaPlayer = new MediaPlayer();
            this.mediaPlayer = mediaPlayer;
            mediaPlayer.setDataSource(str);
            this.mediaPlayer.setOnPreparedListener(this);
            this.mediaPlayer.setOnCompletionListener(this);
            this.mediaPlayer.prepareAsync();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void releasePlay() {
        MediaPlayer mediaPlayer = this.mediaPlayer;
        if (mediaPlayer != null) {
            mediaPlayer.release();
            this.mediaPlayer = null;
        }
    }

    public MessageModel.Message getItem(int i) {
        return this.mChatList.get(i);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mChatList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        if (Constans.MESSAGE_TYPE_REMOTE.equals(this.mChatList.get(i).getType()) && this.mChatList.get(i).getAndroidSend() == 1) {
            return 3;
        }
        if (Constans.MESSAGE_TYPE_REMOTE.equals(this.mChatList.get(i).getType()) && this.mChatList.get(i).getAndroidSend() == 0) {
            return 4;
        }
        if (this.mChatList.get(i).getRevoke() == 1) {
            return 2;
        }
        return this.mChatList.get(i).getAndroidSend() == 1 ? 1 : 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        LogUtil.e(String.format("消息%d, name:%s, content:%s, localContent:%s type:%s", Integer.valueOf(i), this.mChatList.get(i).getAlias(), this.mChatList.get(i).getContent(), this.mChatList.get(i).getLocalPath(), this.mChatList.get(i).getType()));
        ChatAttachPopup chatAttachPopup = this.mChatAttachPopup;
        if (chatAttachPopup != null) {
            chatAttachPopup.dissPop();
        }
        viewHolder.mItemTxtSendTime.setVisibility(8);
        try {
            if (i > 0) {
                Date parse = this.sdf.parse(this.mChatList.get(i).getDate());
                Date parse2 = this.sdf.parse(this.mChatList.get(i - 1).getDate());
                if (parse != null && parse2 != null && parse.getTime() - parse2.getTime() > 300000) {
                    viewHolder.mItemTxtSendTime.setText(TimeUtil.utc2Local(this.mChatList.get(i).getDate()));
                    viewHolder.mItemTxtSendTime.setVisibility(0);
                }
            } else {
                viewHolder.mItemTxtSendTime.setText(TimeUtil.utc2Local(this.mChatList.get(i).getDate()));
                viewHolder.mItemTxtSendTime.setVisibility(0);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        if (getItemViewType(i) == 3 || getItemViewType(i) == 4) {
            notifyItemReject(viewHolder, i);
            if (this.mChatList.get(i).getAndroidSend() != 1) {
                Glide.with(this.mContext).load(this.mChatList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mItemImgHead);
                viewHolder.mItemImgHead.setKing(this.mChatList.get(i).getKing(), this.mContext, this.mChatList.get(i).getGradeIcon());
            } else if (MyApplication.getApplication().getmUserModel() != null) {
                Glide.with(this.mContext).load(MyApplication.getApplication().getmUserModel().getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mItemImgHead);
                viewHolder.mItemImgHead.setKing(MyApplication.getApplication().getmUserModel().getKing(), this.mContext, MyApplication.getApplication().getmUserModel().getGradeIcon());
            }
            viewHolder.mItemImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getAndroidSend() == 1) {
                        ChatAdapter.this.mContext.startActivity(new Intent(ChatAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                    } else if (ChatAdapter.this.clickCallback != null) {
                        ChatAdapter.this.clickCallback.onClickUser(i);
                    }
                }
            });
            return;
        }
        if (this.mChatList.get(i).getRevoke() != 0) {
            if (this.mChatList.get(i).getRevoke() == 1) {
                int androidSend = this.mChatList.get(i).getAndroidSend();
                Context context = this.mContext;
                if (androidSend == 1) {
                    LanguageUtil.setText(context, viewHolder.mItemTxtRevoke, "uRevokeMessage");
                    return;
                } else {
                    viewHolder.mItemTxtRevoke.setText(LanguageUtil.getLanguageByKey(context, "revokeMessage").replace("***", this.mChatList.get(i).getAlias()));
                    return;
                }
            }
            return;
        }
        if (viewHolder.chat_layout_reference != null) {
            viewHolder.chat_layout_reference.setVisibility(8);
        }
        OnRevokeClickListener onRevokeClickListener = new OnRevokeClickListener(i);
        viewHolder.mItemTxtContent.setOnLongClickListener(onRevokeClickListener);
        viewHolder.mItemImgContent.setOnLongClickListener(onRevokeClickListener);
        viewHolder.mItemLayoutVoice.setOnLongClickListener(onRevokeClickListener);
        if (this.mChatList.get(i).getAndroidSend() != 1) {
            Glide.with(this.mContext).load(this.mChatList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mItemImgHead);
            viewHolder.mItemImgHead.setKing(this.mChatList.get(i).getKing(), this.mContext, this.mChatList.get(i).getGradeIcon());
        } else if (MyApplication.getApplication().getmUserModel() != null) {
            Glide.with(this.mContext).load(MyApplication.getApplication().getmUserModel().getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mItemImgHead);
            viewHolder.mItemImgHead.setKing(MyApplication.getApplication().getmUserModel().getKing(), this.mContext, MyApplication.getApplication().getmUserModel().getGradeIcon());
        }
        viewHolder.mItemImgContent.setImageBitmap(null);
        if ("image".equals(this.mChatList.get(i).getType())) {
            viewHolder.mItemTxtContent.setVisibility(8);
            viewHolder.mItemImgContent.setVisibility(0);
            viewHolder.mItemLayoutVoice.setVisibility(8);
            new RequestOptions();
            RequestOptions bitmapTransform = RequestOptions.bitmapTransform(new RoundedCorners(30));
            String content = this.mChatList.get(i).getContent();
            if (TextUtils.isEmpty(content)) {
                content = this.mChatList.get(i).getLocalPath();
            }
            ProgressManager.addListener(content, new OnProgressListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.2
                @Override // com.hismith.hismithapp.image_preview.glide.progress.OnProgressListener
                public void onProgress(String str, boolean z, int i2, long j, long j2) {
                    LogUtil.e("url:" + str + ", precent:" + i2 + ", bytesRead:" + j + ", totalBytes:" + j2);
                }
            });
            Glide.with(this.mContext).asBitmap().load(content).apply((BaseRequestOptions<?>) bitmapTransform).into((RequestBuilder<Bitmap>) new SimpleTarget<Bitmap>() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.3
                @Override // com.bumptech.glide.request.target.BaseTarget, com.bumptech.glide.request.target.Target
                public void onLoadFailed(Drawable drawable) {
                    super.onLoadFailed(drawable);
                    viewHolder.mItemImgContent.setImageResource(R.mipmap.ic_info_error);
                }

                public void onResourceReady(Bitmap bitmap, Transition<? super Bitmap> transition) {
                    int width = bitmap.getWidth();
                    int height = bitmap.getHeight();
                    int dp2px = ChatAdapter.this.dp2px(200);
                    int i2 = (height * dp2px) / width;
                    ViewGroup.LayoutParams layoutParams = viewHolder.mItemImgContent.getLayoutParams();
                    layoutParams.height = i2;
                    layoutParams.width = dp2px;
                    viewHolder.mItemImgContent.setImageBitmap(bitmap);
                    viewHolder.mItemImgContent.setTag(Integer.valueOf(i));
                    if (i == ChatAdapter.this.mChatList.size() - 1 && ChatAdapter.this.isFirstLoad) {
                        ChatAdapter.this.isFirstLoad = false;
                        ChatAdapter.this.mLayoutManager.scrollToPositionWithOffset(i, Integer.MIN_VALUE);
                    }
                }

                @Override // com.bumptech.glide.request.target.Target
                public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                    onResourceReady((Bitmap) obj, (Transition<? super Bitmap>) transition);
                }
            });
            viewHolder.mItemImgContent.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.4
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    LogUtil.e("img onClick");
                    if (view.getTag() == null) {
                        return;
                    }
                    ArrayList arrayList = new ArrayList();
                    int i2 = 0;
                    for (int i3 = 0; i3 < ChatAdapter.this.mChatList.size(); i3++) {
                        if ("image".equals(((MessageModel.Message) ChatAdapter.this.mChatList.get(i3)).getType()) && ((MessageModel.Message) ChatAdapter.this.mChatList.get(i3)).getRevoke() == 0) {
                            String content2 = ((MessageModel.Message) ChatAdapter.this.mChatList.get(i3)).getContent();
                            if (TextUtils.isEmpty(content2)) {
                                content2 = ((MessageModel.Message) ChatAdapter.this.mChatList.get(i3)).getLocalPath();
                            }
                            arrayList.add(content2);
                            if (i3 == ((Integer) view.getTag()).intValue()) {
                                i2 = arrayList.size() - 1;
                            }
                        }
                    }
                    ImagePreview.getInstance().setContext(ChatAdapter.this.mContext).setIndex(i2).setImageList(arrayList).setFolderName("hismith/image").setEnableDragClose(true).start();
                }
            });
        } else if ("voice".equals(this.mChatList.get(i).getType())) {
            viewHolder.mItemTxtContent.setVisibility(8);
            viewHolder.mItemImgContent.setVisibility(8);
            viewHolder.mItemLayoutVoice.setVisibility(0);
            viewHolder.mItemTxtTime.setText(this.mChatList.get(i).getVoiceLength() + "''");
        } else {
            viewHolder.mItemTxtContent.setVisibility(0);
            viewHolder.mItemImgContent.setVisibility(8);
            viewHolder.mItemLayoutVoice.setVisibility(8);
            if (isHttpUrl(this.mChatList.get(i).getContent())) {
                callService(this.mChatList.get(i).getContent(), viewHolder.mItemTxtContent);
            } else {
                viewHolder.mItemTxtContent.setText(this.mChatList.get(i).getContent());
            }
        }
        viewHolder.mItemImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getAndroidSend() == 1) {
                    ChatAdapter.this.mContext.startActivity(new Intent(ChatAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                } else if (ChatAdapter.this.clickCallback != null) {
                    ChatAdapter.this.clickCallback.onClickUser(i);
                }
            }
        });
        if (this.mChatList.get(i).getStatus() == 0) {
            viewHolder.mItemProgress.setVisibility(4);
            viewHolder.mItemImgError.setVisibility(4);
        } else if (this.mChatList.get(i).getStatus() == 1) {
            viewHolder.mItemProgress.setVisibility(0);
            viewHolder.mItemImgError.setVisibility(4);
        } else if (this.mChatList.get(i).getStatus() == 2 || this.mChatList.get(i).getStatus() == 3) {
            viewHolder.mItemProgress.setVisibility(4);
            viewHolder.mItemImgError.setVisibility(0);
        }
        viewHolder.mItemImgError.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                int status = ((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getStatus();
                ChatAdapter chatAdapter = ChatAdapter.this;
                if (status == 2) {
                    ((ChatActivity) chatAdapter.mContext).msgSend(((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getType(), ((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getContent(), i);
                } else if (((MessageModel.Message) chatAdapter.mChatList.get(i)).getStatus() == 3) {
                    ((ChatActivity) ChatAdapter.this.mContext).uploadFile(((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getType(), ((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).getLocalPath(), i);
                }
                ((MessageModel.Message) ChatAdapter.this.mChatList.get(i)).setStatus(1);
                ChatAdapter.this.notifyDataSetChanged();
            }
        });
        viewHolder.mItemLayoutVoice.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatAdapter.7
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                LogUtil.e("voice onClick");
                int i2 = ChatAdapter.this.playPosition;
                int i3 = i;
                ChatAdapter chatAdapter = ChatAdapter.this;
                if (i2 == i3) {
                    chatAdapter.playPosition = -1;
                    ChatAdapter.this.releasePlay();
                } else {
                    chatAdapter.playPosition = i3;
                    ChatAdapter chatAdapter2 = ChatAdapter.this;
                    chatAdapter2.playVoice(((MessageModel.Message) chatAdapter2.mChatList.get(i)).getContent());
                }
                ChatAdapter.this.notifyDataSetChanged();
            }
        });
        if (this.mChatList.get(i).getAndroidSend() == 1) {
            viewHolder.mItemImgPlay.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_chat_voice_right_3));
        } else {
            viewHolder.mItemImgPlay.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_chat_voice_left_3));
        }
        if (this.playPosition == i) {
            if (this.mChatList.get(i).getAndroidSend() == 1) {
                viewHolder.mItemImgPlay.setImageResource(R.drawable.ic_chat_voice_play_right);
            } else {
                viewHolder.mItemImgPlay.setImageResource(R.drawable.ic_chat_voice_play_left);
            }
            ((AnimationDrawable) viewHolder.mItemImgPlay.getDrawable()).start();
        }
        if (this.mChatList.get(i).getReferenceId() > 0) {
            viewHolder.chat_layout_reference.setVisibility(0);
            ReferenceMsg referenceMsg = this.mChatList.get(i).getReferenceMsg();
            if (referenceMsg == null) {
                viewHolder.item_reference_layout_voice.setVisibility(8);
                viewHolder.item_reference_image.setVisibility(8);
                LanguageUtil.setText(this.mContext, viewHolder.item_reference_text, "referenceNotExist");
                return;
            }
            if ("image".equals(referenceMsg.getType())) {
                viewHolder.item_reference_layout_voice.setVisibility(8);
                viewHolder.item_reference_image.setVisibility(0);
                viewHolder.item_reference_text.setText(String.format("%s:", referenceMsg.getUserName()));
                String localPath = referenceMsg.getLocalPath();
                if (TextUtils.isEmpty(localPath)) {
                    localPath = referenceMsg.getContent();
                } else if (!PictureMimeType.isContent(localPath) && !new File(localPath).exists()) {
                    localPath = referenceMsg.getContent();
                }
                LogUtil.e("内容：" + this.mChatList.get(i).getContent() + ", 引用图片内容：" + localPath);
                Glide.with(this.mContext).asBitmap().load(localPath).placeholder(R.mipmap.hismith_image_loading).error(R.mipmap.hismith_image_load_err).transform(new RoundedCornersTransformation(MeasureUtil.dp2px(this.mContext, 15.0f), 0, RoundedCornersTransformation.CornerType.ALL)).into(viewHolder.item_reference_image);
                viewHolder.item_reference_image.setOnClickListener(new ImagePreviewClickListener(localPath));
                return;
            }
            if ("voice".equals(referenceMsg.getType())) {
                viewHolder.item_reference_layout_voice.setVisibility(0);
                viewHolder.item_reference_image.setVisibility(8);
                viewHolder.item_reference_text.setText(String.format("%s:", referenceMsg.getUserName()));
                viewHolder.item_reference_txt_time.setText(String.format("%d''", Integer.valueOf(referenceMsg.getVoiceLength())));
                viewHolder.item_reference_layout_voice.setOnClickListener(new VoicePlayClickListener(referenceMsg.getReferenceId(), referenceMsg.getContent()));
                return;
            }
            if ("text".equals(referenceMsg.getType())) {
                viewHolder.item_reference_layout_voice.setVisibility(8);
                viewHolder.item_reference_image.setVisibility(8);
                viewHolder.item_reference_text.setText(String.format("%s:%s", referenceMsg.getUserName(), referenceMsg.getContent()));
            } else {
                viewHolder.item_reference_layout_voice.setVisibility(8);
                viewHolder.item_reference_image.setVisibility(8);
                viewHolder.item_reference_text.setText("[Message Unsupported]");
            }
        }
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        LogUtil.e("adapter position:" + i + ", status:" + this.mChatList.get(i).getStatus());
        if (getItemViewType(i) == 3 || getItemViewType(i) == 4) {
            notifyItemReject(viewHolder, i);
            return;
        }
        int revoke = this.mChatList.get(i).getRevoke();
        List<MessageModel.Message> list2 = this.mChatList;
        if (revoke != 0) {
            if (list2.get(i).getRevoke() == 1) {
                if (this.mChatList.get(i).getAndroidSend() == 1) {
                    if (viewHolder.mItemTxtRevoke != null) {
                        LanguageUtil.setText(this.mContext, viewHolder.mItemTxtRevoke, "uRevokeMessage");
                        return;
                    }
                    return;
                } else {
                    String replace = LanguageUtil.getLanguageByKey(this.mContext, "revokeMessage").replace("***", this.mChatList.get(i).getAlias());
                    if (viewHolder.mItemTxtRevoke != null) {
                        viewHolder.mItemTxtRevoke.setText(replace);
                        return;
                    }
                    return;
                }
            }
            return;
        }
        if (list2.get(i).getStatus() == 0) {
            if (viewHolder.mItemProgress != null) {
                viewHolder.mItemProgress.setVisibility(4);
            }
            if (viewHolder.mItemImgError != null) {
                viewHolder.mItemImgError.setVisibility(4);
                return;
            }
            return;
        }
        if (this.mChatList.get(i).getStatus() == 1) {
            if (viewHolder.mItemProgress != null) {
                viewHolder.mItemProgress.setVisibility(0);
            }
            if (viewHolder.mItemImgError != null) {
                viewHolder.mItemImgError.setVisibility(4);
                return;
            }
            return;
        }
        if (this.mChatList.get(i).getStatus() == 2 || this.mChatList.get(i).getStatus() == 3) {
            if (viewHolder.mItemProgress != null) {
                viewHolder.mItemProgress.setVisibility(4);
            }
            if (viewHolder.mItemImgError != null) {
                viewHolder.mItemImgError.setVisibility(0);
            }
        }
    }

    @Override // android.media.MediaPlayer.OnCompletionListener
    public void onCompletion(MediaPlayer mediaPlayer) {
        this.playPosition = -1;
        releasePlay();
        notifyDataSetChanged();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return i == 4 ? new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_chat_remote_left, viewGroup, false)) : i == 3 ? new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_chat_remote_right, viewGroup, false)) : i == 2 ? new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_chat_revoke, viewGroup, false)) : i == 1 ? new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_chat_right, viewGroup, false)) : new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_chat_left, viewGroup, false));
    }

    @Override // android.media.MediaPlayer.OnPreparedListener
    public void onPrepared(MediaPlayer mediaPlayer) {
        try {
            mediaPlayer.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setClickCallback(ClickCallback clickCallback) {
        this.clickCallback = clickCallback;
    }

    public void setOnActionClickListener(OnActionClickListener onActionClickListener) {
        this.onActionClickListener = onActionClickListener;
    }
}
