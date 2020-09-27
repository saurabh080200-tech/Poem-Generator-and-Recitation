import streamlit as st
import tensorflow as tf
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import pyttsx3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional
from tensorflow.keras.optimizers import Adam
engine=pyttsx3.init()
engine.setProperty('rate',150)

def main():
    st.title("Poem Generator 📚")
    st.sidebar.title("Lets Get Started....📖")

    select=st.sidebar.selectbox("Choose An Option",("About","Poem Generation"),key="select")

    if select=="About":
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.markdown("### This Web Application is designed to generate some line of poetry.")
        st.markdown("### It uses power LSTM network which learn sequence of tokenizeed text")
        st.text(" ")
        st.markdown("### Care To Take A look..")

    if select=="Poem Generation":
        st.sidebar.subheader("Select the Type of Poem You want To Generate")
        choose=st.sidebar.selectbox("Select The Type",("Funny","Nature","Romantic","Inspirational","For Children"),key="choose")

        if choose=="Funny":
            st.text(" ")
            st.text(" ")
            st.markdown("We Have 2 Different Types of %s Poems." %(choose))
            st.markdown("Please Select The following Poem You wish your poem to be like..")

            poem_selection=st.selectbox("Select the Poem",("Cinderella","Television"),key="poem_selection")
            if poem_selection=="Cinderella":
                with open("Funny Cinderella.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See the Cinderella Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t1")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input1=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input1=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input1)*2):
                    if user_input1==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input1])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input1.split()
                    if len(split)%6==0:
                        user_input1 +="\n"+output_word
                    else:
                        user_input1 += " " + output_word
                
                if user_input1!=" ":
                    st.write(user_input1)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input1)
                        engine.runAndWait()

            if poem_selection=="Television":
                with open("Funny Television.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See the Television Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t3")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input2=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input2=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input2)*2):
                    if user_input2==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input2])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input2.split()
                    if len(split)%6==0:
                        user_input2 +="\n"+output_word
                    else:
                        user_input2 += " " + output_word
                
                if user_input2!=" ":
                    st.write(user_input2)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input2)
                        engine.runAndWait()

        if choose=="For Children":
            st.text(" ")
            st.text(" ")
            st.markdown("We Have 2 Different Types of Poems for %s." %(choose))
            st.markdown("Please Select The following Poem You wish your poem to be like..")

            poem_selection=st.selectbox("Select the Poem",("A Worm In My Pocket","Friends"),key="poem_selection")
            if poem_selection=="A Worm In My Pocket":
                with open("For Children A Worm In My Pocket.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See the A Worm In My Pocket Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t6")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input3=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input3=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input3)*2):
                    if user_input3==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input3])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input3.split()
                    if len(split)%6==0:
                        user_input3 +="\n"+output_word
                    else:
                        user_input3 += " " + output_word
                
                if user_input3!=" ":
                    st.write(user_input3)
                    
                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input3)
                        engine.runAndWait()
                    
            if poem_selection=="Friends":
                with open("For Children Friends.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See the Friends Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t8")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input4=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input4=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input)*2):
                    if user_input4==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input4])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input4.split()
                    if len(split)%6==0:
                        user_input4 +="\n"+output_word
                    else:
                        user_input4 += " " + output_word
                
                if user_input4!=" ":
                    st.write(user_input4)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input4)
                        engine.runAndWait()

        
        if choose=="Inspirational":
            st.text(" ")
            st.text(" ")
            st.markdown("We Have 2 Different Types of %s Poems." %(choose))
            st.markdown("Please Select The following Poem You wish your poem to be like..")

            poem_selection=st.selectbox("Select the Poem",("My Personal Quest","Pain Ends"),key="poem_selection")
            if poem_selection=="My Personal Quest":
                with open("Inspirational My Personal Quest.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See the My Personal Quest Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t11")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input5=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input5=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input5)*2):
                    if user_input5==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input5])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input5.split()
                    if len(split)%6==0:
                        user_input5 +="\n"+output_word
                    else:
                        user_input5 += " " + output_word
                
                if user_input5!=" ":
                    st.write(user_input5)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input5)
                        engine.runAndWait()


            if poem_selection=="Pain Ends":
                with open("Inspirational Pains Ends.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See the Pain Ends Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t12")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input6=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input6=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input6)*2):
                    if user_input6==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input6])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input6.split()
                    if len(split)%6==0:
                        user_input6 +="\n"+output_word
                    else:
                        user_input6 += " " + output_word
                
                if user_input6!=" ":
                    st.write(user_input6)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input6)
                        engine.runAndWait()


        if choose=="Nature":
            st.text(" ")
            st.text(" ")
            st.markdown("We Have 2 Different Types of %s Poems." %(choose))
            st.markdown("Please Select The following Poem You wish your poem to be like..")

            poem_selection=st.selectbox("Select the Poem",("I Wandered Lonely As A Cloud","Stopping By Woods On A Snowy Evening"),key="poem_selection")
            if poem_selection=="I Wandered Lonely As A Cloud":
                with open("Nature I Wandered Lonely As A Cloud.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See I Wandered Lonely As A Cloud Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t16")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input7=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input7=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input7)*2):
                    if user_input7==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input7])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input7.split()
                    if len(split)%6==0:
                        user_input7 +="\n"+output_word
                    else:
                        user_input7 += " " + output_word
                
                if user_input7!=" ":
                    st.write(user_input7)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input7)
                        engine.runAndWait()

            if poem_selection=="Stopping By Woods On A Snowy Evening":
                with open("Nature Stopping By Woods On A Snowy Evening.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See Stopping By Woods On A Snowy Evening Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t18")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input8=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input8=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input8)*2):
                    if user_input8==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input8])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input8.split()
                    if len(split)%6==0:
                        user_input8 +="\n"+output_word
                    else:
                        user_input8 += " " + output_word
                
                if user_input8!=" ":
                    st.write(user_input8)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input8)
                        engine.runAndWait()

        if choose=="Romantic":
            st.text(" ")
            st.text(" ")
            st.markdown("We Have 2 Different Types of %s Poems." %(choose))
            st.markdown("Please Select The following Poem You wish your poem to be like..")

            poem_selection=st.selectbox("Select the Poem",("Baby When You Hold Me","I Love You"),key="poem_selection")
            if poem_selection=="Baby When You Hold Me":
                with open("Romantic Baby When You Hold Me.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See Baby When You Hold Me Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t21")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input9=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input9=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input9)*2):
                    if user_input9==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input9])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input9.split()
                    if len(split)%6==0:
                        user_input9 +="\n"+output_word
                    else:
                        user_input9 += " " + output_word
                
                if user_input9!=" ":
                    st.write(user_input9)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input9)
                        engine.runAndWait()


            if poem_selection=="I Love You":
                with open("Romantic I Love You.txt","r") as fd:
                    text1=fd.read()
                
                corpus1=text1.split("\n")

                if st.checkbox("Would You Like to See I Love You Poem",False):
                    for i in corpus1:
                        st.write(i)

                @st.cache(allow_output_mutation=True)
                def load_model():
                    model=tf.keras.models.load_model("model.t22")
                    return model

                model=load_model()
                
                st.subheader("Please Enter The Starter Text for the Poem")
                user_input10=st.text_input("Enter Here.."," ")

                if st.checkbox("Clear the Output??"):
                    user_input10=" "

                tokenizer=Tokenizer()
                tokenizer.fit_on_texts(corpus1)
                total_words=len(tokenizer.word_index)+1

                input_sequences=[]
                for line in corpus1:
                    token_list=tokenizer.texts_to_sequences([line])[0]
                    for i in range(1,len(token_list)):
                        n_gram=token_list[:i+1]
                        input_sequences.append(n_gram)

                max_sequence_length=max([len(x) for x in input_sequences])

                for _ in range(len(user_input10)*2):
                    if user_input10==" ":
                        break
                    token_list = tokenizer.texts_to_sequences([user_input10])[0]
                    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
                    predicted = np.argmax(model.predict(token_list),axis=-1)
                    output_word = ""
                    for word, index in tokenizer.word_index.items():
                        if index == predicted:
                            output_word = word
                            break
            
                    split=user_input10.split()
                    if len(split)%6==0:
                        user_input10 +="\n"+output_word
                    else:
                        user_input10 += " " + output_word
                
                if user_input10!=" ":
                    st.write(user_input10)

                    if st.checkbox("Want to Listen to the Poem You Generated.."):
                        engine.say(user_input10)
                        engine.runAndWait()

if __name__=="__main__":
    main()