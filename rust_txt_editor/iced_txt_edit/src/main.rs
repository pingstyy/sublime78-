use iced::{Sandbox , Element, Settings};
use iced::widget::{container , text_editor } ;

fn main() -> iced::Result{
    Editor::run(Settings::default())
}

struct Editor{
    content: text_editor::Content ,
}

#[derive(Debug, Clone)]

enum Message  {
    Edit(text_editor::Action) ,
}

impl Sandbox for Editor{
    type Message = Message ;

    fn new() -> Self{
        Self{
            content : text_editor::Content::new(),
        }
    }


    fn title(&self) -> String{
        String::from("Editex")
    }


    fn update(&mut self, message: Message){
        match message {
            Message::Edit(action) =>{
                self.content.edit(action);
            }
        }
    }
    
    fn view(&self) -> Element<'_, Self::Message>{
        let inpt = text_editor(&self.content).on_edit(Message::Edit).into() ;
        container(inpt).padding(10).into() 
    }



}
