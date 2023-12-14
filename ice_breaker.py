from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatGooglePalm
from langchain.chains import LLMChain
import os


text_description = "Sir Ridley Scott (born 30 November 1937) is an English filmmaker. He is best known for directing films in the science fiction, crime, and historical drama genres. His work is known for its atmospheric and highly concentrated visual style.[1][2][3] Scott has received many accolades, including the BAFTA Fellowship for lifetime achievement in 2018, two Primetime Emmy Awards, and a Golden Globe Award.[4] In 2003, he was knighted by Queen Elizabeth II.[5]\
An alumnus of the Royal College of Art in London, Scott began his career in television as a designer and director before moving into advertising as a director of commercials. He made his film directorial debut with The Duellists (1977) and gained wider recognition with his next film, Alien (1979). Though his films range widely in setting and period, they showcase memorable imagery of urban environments, spanning 2nd-century Rome in Gladiator (2000), 12th-century Jerusalem in Kingdom of Heaven (2005), medieval England in Robin Hood (2010), ancient Memphis in Exodus: Gods and Kings (2014), contemporary Mogadishu in Black Hawk Down (2001), and the futuristic cityscapes of Blade Runner (1982) and different planets in Alien, Prometheus (2012), The Martian (2015) and Alien: Covenant (2017). Several of his films are also known for their strong female characters, such as Alien, Thelma & Louise (1991) and Napoleon (2023).[6][7]\
Scott has been nominated for three Academy Awards for Directing for Thelma & Louise, Gladiator and Black Hawk Down.[2] Gladiator won the Academy Award for Best Picture, and he received a nomination in the same category for The Martian. In 1995, both Scott and his brother Tony received a British Academy Film Award for Outstanding British Contribution to Cinema.[8] In a 2004 BBC poll, Scott was ranked 10 on the list of most influential people in British culture.[9] Scott is also known for his work in television, having earned 10 Primetime Emmy Award nominations. He won twice, for Outstanding Television Film for the HBO film The Gathering Storm (2002) and for Outstanding Documentary or Nonfiction Special for the History Channel's Gettysburg (2011).[10] He was Emmy-nominated for RKO 281 (1999), The Andromeda Strain (2008), and The Pillars of the Earth (2010).[11]\
"

if __name__ == '__main__':
    print("hello langchain!")
    print(os.getenv('OPENAI_API_KEY'))

summary_template=" given the LinkedIn information {information} about a person I want you to create:\
    1. a short summary\
    2. two interesting facts about them\
    "

summary_prompt_template   = PromptTemplate.from_template(summary_template)
#summary_prompt_template.format(information=text_description)

openai_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chain = LLMChain(llm=openai_llm,prompt=summary_prompt_template)
print(chain.run(information=text_description))