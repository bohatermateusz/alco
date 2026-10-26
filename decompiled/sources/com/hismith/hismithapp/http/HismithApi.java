package com.hismith.hismithapp.http;

import com.hismith.hismithapp.model.BarrageModel;
import com.hismith.hismithapp.model.BlackModel;
import com.hismith.hismithapp.model.BlockModel;
import com.hismith.hismithapp.model.ChatUpModel;
import com.hismith.hismithapp.model.CheckUpdateModel;
import com.hismith.hismithapp.model.ComplaintReasonsModel;
import com.hismith.hismithapp.model.CountryModel;
import com.hismith.hismithapp.model.FansFocusModel;
import com.hismith.hismithapp.model.FirmwareModel;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.model.InteractiveModel;
import com.hismith.hismithapp.model.LikeUserModel;
import com.hismith.hismithapp.model.LoginModel;
import com.hismith.hismithapp.model.MessageModel;
import com.hismith.hismithapp.model.ModeBlockModel;
import com.hismith.hismithapp.model.MsgSendModel;
import com.hismith.hismithapp.model.PlayLogsListModel;
import com.hismith.hismithapp.model.PlayLogsModel;
import com.hismith.hismithapp.model.PlayModeActionModel;
import com.hismith.hismithapp.model.PlayModeCreateModel;
import com.hismith.hismithapp.model.PlayModeListModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.ProductsModel;
import com.hismith.hismithapp.model.ReviewDetailsModel;
import com.hismith.hismithapp.model.ReviewLikeModel;
import com.hismith.hismithapp.model.ReviewSendModel;
import com.hismith.hismithapp.model.TokenModel;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicCategoriesModel;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.model.TopicReviewModel;
import com.hismith.hismithapp.model.TopicSendModel;
import com.hismith.hismithapp.model.UploadModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.utils.Constans;
import io.reactivex.Observable;
import java.util.Map;
import okhttp3.RequestBody;
import okhttp3.ResponseBody;
import retrofit2.http.Field;
import retrofit2.http.FormUrlEncoded;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.PartMap;
import retrofit2.http.Streaming;
import retrofit2.http.Url;

/* loaded from: classes3.dex */
public interface HismithApi {
    @POST("allProducts_v1")
    Observable<ProductsModel> allProducts_v1();

    @FormUrlEncoded
    @POST("changeMail")
    Observable<HttpModel> changeMail(@Field("token") String str, @Field("email") String str2, @Field("code") Integer num);

    @FormUrlEncoded
    @POST("changePass")
    Observable<HttpModel> changePass(@Field("token") String str, @Field("oldPassword") String str2, @Field("password") String str3);

    @FormUrlEncoded
    @POST("checkUpdate")
    Observable<CheckUpdateModel> checkUpdate(@Field("token") String str, @Field("allLangs") String str2, @Field("dev") Integer num);

    @FormUrlEncoded
    @POST("checkUpdate")
    Observable<CheckUpdateModel> checkUpdate(@Field("token") String str, @Field("allLangs") String str2, @Field("dev") Integer num, @Field("type") String str3, @Field("version") String str4);

    @FormUrlEncoded
    @POST("checkUpdate")
    Observable<CheckUpdateModel> checkUpdate(@Field("token") String str, @Field("allLangs") String str2, @Field("dev") Integer num, @Field("type") String str3, @Field("version") String str4, @Field("ads") String str5);

    @FormUrlEncoded
    @POST("complaint")
    Observable<HttpModel> complaint(@Field("token") String str, @Field("id") Integer num, @Field("type") String str2, @Field("reason") String str3, @Field("content") String str4);

    @FormUrlEncoded
    @POST(Constans.complaintReasons)
    Observable<ComplaintReasonsModel> complaintReasons(@Field("token") String str);

    @Streaming
    @POST
    Observable<ResponseBody> download(@Url String str);

    @FormUrlEncoded
    @POST("emailVerCode")
    Observable<HttpModel> emailVerCode(@Field("type") String str, @Field("email") String str2);

    @FormUrlEncoded
    @POST("emailVerCode")
    Observable<HttpModel> emailVerCode(@Field("token") String str, @Field("type") String str2, @Field("email") String str3);

    @FormUrlEncoded
    @POST("firmwareVersion")
    Observable<FirmwareModel> firmwareVersion(@Field("token") String str, @Field("action") String str2, @Field("productId") String str3);

    @FormUrlEncoded
    @POST("follow")
    Observable<FollowModel> follow(@Field("token") String str, @Field("userId") Integer num, @Field("follow") Integer num2);

    @FormUrlEncoded
    @POST("forgotPass")
    Observable<LoginModel> forgotPass(@Field("email") String str, @Field("password") String str2, @Field("code") Integer num);

    @GET("getCountry")
    Observable<CountryModel> getCountry();

    @FormUrlEncoded
    @POST("getFansFocus")
    Observable<FansFocusModel> getFansFocus(@Field("token") String str, @Field("mode") String str2, @Field("page") Integer num);

    @FormUrlEncoded
    @POST("getUserInfo")
    Observable<UserModel> getMyInfo(@Field("token") String str, @Field("androidToken") String str2, @Field("appVersion") String str3);

    @FormUrlEncoded
    @POST("playModeBarrage?action=get")
    Observable<BarrageModel> getPlayModeBarrage(@Field("token") String str, @Field("id") int i);

    @FormUrlEncoded
    @POST("getProduct")
    Observable<ProductModel> getProduct(@Field("token") String str, @Field("code") String str2);

    @FormUrlEncoded
    @POST("getUserBlackList")
    Observable<BlackModel> getUserBlackList(@Field("token") String str, @Field("page") Integer num);

    @FormUrlEncoded
    @POST("getUserInfo")
    Observable<UserModel> getUserInfo(@Field("token") String str, @Field("userId") Integer num);

    @FormUrlEncoded
    @POST("userLogin")
    Observable<LoginModel> login(@Field("userName") String str, @Field("userPass") String str2);

    @FormUrlEncoded
    @POST("userLogin")
    Observable<LoginModel> login(@Field("userName") String str, @Field("userPass") String str2, @Field("loginType") String str3);

    @FormUrlEncoded
    @POST("messageNew")
    Observable<MessageModel> messageNew(@Field("token") String str, @Field("lastMsgId") Integer num);

    @FormUrlEncoded
    @POST("messageNew_v1")
    Observable<MessageModel> messageNew_v1(@Field("token") String str, @Field("lastMsgId") Integer num);

    @FormUrlEncoded
    @POST("msgAction")
    Observable<HttpModel> msgAction(@Field("token") String str, @Field("msgId") int i, @Field("action") String str2);

    @FormUrlEncoded
    @POST("msgFirstChat")
    Observable<ChatUpModel> msgFirstChat(@Field("token") String str, @Field("page") Integer num);

    @FormUrlEncoded
    @POST("msgInteractive")
    Observable<InteractiveModel> msgInteractive(@Field("token") String str, @Field("page") Integer num);

    @FormUrlEncoded
    @POST("msgSend")
    Observable<MsgSendModel> msgSend(@Field("token") String str, @Field("localId") String str2, @Field("userId") Integer num, @Field("content") String str3, @Field("voiceLength") Integer num2, @Field("type") String str4);

    @FormUrlEncoded
    @POST("msgSend")
    Observable<MsgSendModel> msgSend(@Field("token") String str, @Field("localId") String str2, @Field("userId") Integer num, @Field("content") String str3, @Field("voiceLength") Integer num2, @Field("type") String str4, @Field("referenceId") Integer num3);

    @FormUrlEncoded
    @POST("userTopic")
    Observable<TopicListModel> myTopic(@Field("token") String str, @Field("type") String str2, @Field("page") Integer num);

    @FormUrlEncoded
    @POST("playLogs")
    Observable<PlayLogsListModel> playLogs(@Field("token") String str, @Field("new") int i, @Field("requestId") String str2);

    @FormUrlEncoded
    @POST("playModeAction")
    Observable<PlayModeActionModel> playModeAction(@Field("token") String str, @Field("id") int i, @Field("action") String str2);

    @FormUrlEncoded
    @POST("playModeAction")
    Observable<PlayModeActionModel> playModeAction(@Field("token") String str, @Field("id") int i, @Field("action") String str2, @Field("name") String str3);

    @FormUrlEncoded
    @POST("playModeBlock")
    Observable<ModeBlockModel> playModeBlock(@Field("token") String str, @Field("requestId") String str2);

    @FormUrlEncoded
    @POST("playModeList")
    Observable<PlayModeListModel> playModeList(@Field("token") String str, @Field("type") String str2, @Field("new") int i, @Field("requestId") String str3, @Field("sortModeTime") String str4, @Field("sortPlayCount") String str5, @Field("filterOnlyThisProduct") int i2, @Field("filterModeName") String str6, @Field("filterModeTime") String str7, @Field("filterModeSpeed") int i3, @Field("filterModePlayCount") String str8, @Field("filterPublishDate") String str9);

    @FormUrlEncoded
    @POST("playModeList")
    Observable<PlayModeListModel> playModeList(@Field("token") String str, @Field("type") String str2, @Field("code") String str3, @Field("new") int i, @Field("requestId") String str4);

    @FormUrlEncoded
    @POST("playModeList")
    Observable<PlayModeListModel> playModeList(@Field("token") String str, @Field("type") String str2, @Field("code") String str3, @Field("new") int i, @Field("requestId") String str4, @Field("sortModeTime") String str5, @Field("sortPlayCount") String str6, @Field("filterOnlyThisProduct") int i2, @Field("filterModeName") String str7, @Field("filterModeTime") String str8, @Field("filterModeSpeed") int i3, @Field("filterModePlayCount") String str9, @Field("filterPublishDate") String str10);

    @FormUrlEncoded
    @POST("playModeList")
    Observable<PlayModeListModel> playModeList(@Field("token") String str, @Field("type") String str2, @Field("userId") String str3, @Field("code") String str4, @Field("new") int i, @Field("requestId") String str5);

    @FormUrlEncoded
    @POST("playModeSave")
    Observable<PlayModeCreateModel> playModeSave(@Field("token") String str, @Field("id") String str2, @Field("name") String str3, @Field("code") String str4, @Field("modeData") String str5);

    @FormUrlEncoded
    @POST("playModeSave")
    Observable<PlayModeCreateModel> playModeSave(@Field("token") String str, @Field("id") String str2, @Field("name") String str3, @Field("code") String str4, @Field("setting") String str5, @Field("modeData") String str6);

    @FormUrlEncoded
    @POST
    Observable<HttpModel> qrcodeLogin(@Url String str, @Field("token") String str2);

    @FormUrlEncoded
    @POST("userRegister")
    Observable<LoginModel> register(@Field("email") String str, @Field("password") String str2, @Field("code") Integer num);

    @FormUrlEncoded
    @POST("removeAccount")
    Observable<HttpModel> removeAccount(@Field("token") String str, @Field("code") String str2);

    @FormUrlEncoded
    @POST("follow")
    Observable<FollowModel> removeFan(@Field("token") String str, @Field("userId") Integer num, @Field("follow") Integer num2, @Field("remove") Integer num3);

    @FormUrlEncoded
    @POST("setUserBlackList")
    Observable<BlockModel> removeUserBlackList(@Field("token") String str, @Field("userId") Integer num, @Field("remove") int i);

    @FormUrlEncoded
    @POST("setUserBlackList")
    Observable<BlockModel> removeUserBlackList(@Field("token") String str, @Field("userId") String str2, @Field("remove") int i);

    @FormUrlEncoded
    @POST("oauth2/v4/token")
    Observable<TokenModel> requestToken(@Field("client_id") String str, @Field("client_secret") String str2, @Field("code") String str3, @Field("redirect_uri") String str4, @Field("grant_type") String str5);

    @FormUrlEncoded
    @POST("oauth2/v4/token")
    Observable<TokenModel> requestTokenRefresh(@Field("refresh_token") String str, @Field("client_id") String str2, @Field("client_secret") String str3, @Field("grant_type") String str4);

    @FormUrlEncoded
    @POST
    Observable<ResponseBody> resetServer(@Url String str);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setChatUpState(@Field("token") String str, @Field("chatUpState") Integer num);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setPlayLogs(@Field("token") String str, @Field("playLogs") Integer num);

    @FormUrlEncoded
    @POST("setPlayLogs")
    Observable<PlayLogsModel> setPlayLogs(@Field("token") String str, @Field("productId") String str2, @Field("gameOver") int i);

    @FormUrlEncoded
    @POST("playModeBarrage?action=set")
    Observable<BarrageModel> setPlayModeBarrage(@Field("token") String str, @Field("id") int i, @Field("second") double d, @Field("color") String str2, @Field("content") String str3);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setRemind(@Field("token") String str, @Field("mutuallyRemind") Integer num, @Field("focusRemind") Integer num2, @Field("messageRemind") Integer num3);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setRemind(@Field("token") String str, @Field("mutuallyRemind") Integer num, @Field("focusRemind") Integer num2, @Field("messageRemind") Integer num3, @Field("productAds") Integer num4);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setStealth(@Field("token") String str, @Field("stealth") Integer num);

    @FormUrlEncoded
    @POST("setUserAlias")
    Observable<HttpModel> setUserAlias(@Field("token") String str, @Field("userId") Integer num, @Field("userAlias") String str2);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setUserBIO(@Field("token") String str, @Field("bios") String str2);

    @FormUrlEncoded
    @POST("setUserBlackList")
    Observable<BlockModel> setUserBlackList(@Field("token") String str, @Field("userId") Integer num);

    @FormUrlEncoded
    @POST("setUserBlackList")
    Observable<BlockModel> setUserBlackList(@Field("token") String str, @Field("userId") Integer num, @Field("remove") int i);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setUserHead(@Field("token") String str, @Field("headIcon") String str2);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setUserInfo(@Field("token") String str, @Field("gender") Integer num, @Field("name") String str2, @Field("countryId") Integer num2, @Field("cityId") Integer num3, @Field("phone") String str3, @Field("birthday") String str4, @Field("headIcon") String str5, @Field("voice") String str6, @Field("voiceLength") Integer num4, @Field("photos") String str7, @Field("chatUpState") Integer num5, @Field("focusRemind") Integer num6, @Field("messageRemind") Integer num7, @Field("mutuallyRemind") Integer num8, @Field("addFriendRemind") Integer num9);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setUserInfo(@Field("token") String str, @Field("photos") String str2);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setUserInfo(@Field("token") String str, @Field("name") String str2, @Field("gender") Integer num, @Field("countryId") Integer num2, @Field("cityId") Integer num3, @Field("birthday") String str3, @Field("phone") String str4, @Field("headIcon") String str5);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setUserInfo(@Field("token") String str, @Field("name") String str2, @Field("countryId") Integer num, @Field("cityId") Integer num2, @Field("birthday") String str3, @Field("gender") Integer num3, @Field("interestedIn") Integer num4, @Field("phone") String str4);

    @FormUrlEncoded
    @POST("setUserLang")
    Observable<HttpModel> setUserLang(@Field("token") String str, @Field("userLang") String str2);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setViewSex(@Field("token") String str, @Field("viewSex") int i);

    @FormUrlEncoded
    @POST("setUserInfo")
    Observable<HttpModel> setVoice(@Field("token") String str, @Field("voice") String str2, @Field("voiceLength") Integer num);

    @FormUrlEncoded
    @POST("submitError")
    Observable<HttpModel> submitError(@Field("token") String str, @Field("app") String str2, @Field("appVersion") String str3, @Field("message") String str4);

    @FormUrlEncoded
    @POST("topicAction")
    Observable<TopicActionModel> topicAction(@Field("token") String str, @Field("id") Integer num, @Field("action") String str2);

    @FormUrlEncoded
    @POST("topicCategories")
    Observable<TopicCategoriesModel> topicCategories(@Field("token") String str);

    @FormUrlEncoded
    @POST("topicDel")
    Observable<HttpModel> topicDel(@Field("token") String str, @Field("id") Integer num);

    @FormUrlEncoded
    @POST("topicDetails")
    Observable<TopicModel> topicDetails(@Field("token") String str, @Field("id") Integer num);

    @FormUrlEncoded
    @POST("topicSend")
    Observable<TopicSendModel> topicEdit(@Field("token") String str, @Field("id") Integer num, @Field("localId") String str2, @Field("title") String str3, @Field("content") String str4);

    @FormUrlEncoded
    @POST("topicList")
    Observable<TopicListModel> topicList(@Field("token") String str, @Field("key") String str2, @Field("new") Integer num, @Field("requestId") String str3);

    @FormUrlEncoded
    @POST("topicList")
    Observable<TopicListModel> topicList(@Field("token") String str, @Field("key") String str2, @Field("categorieId") String str3, @Field("new") Integer num, @Field("requestId") String str4);

    @FormUrlEncoded
    @POST("topicReviewDel")
    Observable<HttpModel> topicReviewDel(@Field("token") String str, @Field("id") Integer num);

    @FormUrlEncoded
    @POST("topicReviewDetails")
    Observable<ReviewDetailsModel> topicReviewDetails(@Field("token") String str, @Field("id") Integer num, @Field("page") Integer num2);

    @FormUrlEncoded
    @POST("topicReviewLike")
    Observable<ReviewLikeModel> topicReviewLike(@Field("token") String str, @Field("id") Integer num, @Field("action") String str2);

    @FormUrlEncoded
    @POST("topicReviewLikeUser")
    Observable<LikeUserModel> topicReviewLikeUser(@Field("token") String str, @Field("id") Integer num, @Field("page") Integer num2);

    @FormUrlEncoded
    @POST("topicReviewSend")
    Observable<ReviewSendModel> topicReviewReplySend(@Field("token") String str, @Field("id") Integer num, @Field("content") String str2, @Field("replyId") Integer num2);

    @FormUrlEncoded
    @POST("topicReviewSend")
    Observable<ReviewSendModel> topicReviewSend(@Field("token") String str, @Field("id") Integer num, @Field("content") String str2);

    @FormUrlEncoded
    @POST("topicReviews")
    Observable<TopicReviewModel> topicReviews(@Field("token") String str, @Field("id") Integer num, @Field("page") Integer num2);

    @FormUrlEncoded
    @POST("topicSend")
    Observable<TopicSendModel> topicSend(@Field("token") String str, @Field("localId") String str2, @Field("categorieId") String str3, @Field("title") String str4, @Field("content") String str5);

    @FormUrlEncoded
    @POST("topicSend")
    Observable<TopicSendModel> topicSend(@Field("token") String str, @Field("localId") String str2, @Field("categorieId") String str3, @Field("tag") String str4, @Field("title") String str5, @Field("content") String str6);

    @POST("uploadFile")
    @Multipart
    Observable<UploadModel> uploadFile(@Field("token") String str, @Field("action") String str2, @Field("fileType") String str3, @Field("requestId") String str4, @PartMap Map<String, RequestBody> map);

    @POST("uploadFile")
    @Multipart
    Observable<UploadModel> uploadFile(@PartMap Map<String, RequestBody> map);

    @FormUrlEncoded
    @POST("userOauthLogin")
    Observable<LoginModel> userOauthLogin(@Field("source") String str, @Field("sourceToken") String str2);

    @FormUrlEncoded
    @POST("userOauthLogin")
    Observable<LoginModel> userOauthLogin(@Field("source") String str, @Field("sourceToken") String str2, @Field("version") String str3);

    @FormUrlEncoded
    @POST("userSearch")
    Observable<UserSearchModel> userSearch(@Field("token") String str, @Field("key") String str2, @Field("page") Integer num);

    @FormUrlEncoded
    @POST("userTopic")
    Observable<TopicListModel> userTopic(@Field("token") String str, @Field("userId") String str2, @Field("page") Integer num);
}
