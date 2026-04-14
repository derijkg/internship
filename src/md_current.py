from __future__ import annotations  # IMPORTANT: This fixes the 'type' subscriptable error
import json
from typing import List, Union, Any, Optional
from pydantic import BaseModel
import markdownify
import marko
from marko.block import Heading, Paragraph, List, ListItem

# --- 1. Define the Target AST Schema ---

class ContentBlock(BaseModel):
    type: str  # "text", "list_item", "list_container"
    value: str

class Section(BaseModel):
    type: str = "section"
    title: str
    level: int
    # We use the modern 3.10+ pipe operator '|' and lowercase 'list'
    # This is made possible by 'from __future__ import annotations'
    content: list[ContentBlock | Section]

class Chapter(BaseModel):
    type: str = "chapter"
    title: str
    content: list[ContentBlock | Section]

class DocumentAST(BaseModel):
    title: str
    chapters: list[Chapter]

# Now we tell Pydantic to resolve the circular references (Section referring to Section)
Section.model_rebuild()
Chapter.model_rebuild()
DocumentAST.model_rebuild()

# --- 2. The Converter Engine ---

class MarkerASTConverter:
    def __init__(self):
        self.md_parser = marko.Markdown()
        
    def _get_text(self, node) -> str:
        """
        Robustly extracts text from any marko node by walking the tree.
        This avoids importing specific marko.inline classes.
        """
        if not hasattr(node, 'children'):
            return ""

        if isinstance(node.children, str):
            return node.children
        
        if isinstance(node.children, (list, tuple)):
            return "".join([self._get_text(child) for child in node.children])
        
        return ""

    def convert(self, raw_text: str) -> DocumentAST:
        # Step A: Normalize HTML to Markdown
        clean_markdown = markdownify.markdownify(raw_text, heading_style="ATX")

        # Step B: Parse Markdown to marko AST
        doc = self.md_parser.parse(clean_markdown)

        # Step C: Transform to our Semantic AST
        return self._build_structure(doc)

    def _build_structure(self, doc) -> DocumentAST:
        chapters: list[Chapter] = []
        current_chapter: Optional[Chapter] = None
        current_section: Optional[Section] = None

        for node in doc.children:
            
            # --- HANDLE HEADINGS (Hierarchy Logic) ---
            if isinstance(node, Heading):
                title = self._get_text(node).strip()
                level = node.level

                if level == 1:
                    current_chapter = Chapter(title=title, content=[])
                    chapters.append(current_chapter)
                    current_section = None 
                
                elif level == 2 and current_chapter:
                    current_section = Section(title=title, level=level, content=[])
                    current_chapter.content.append(current_section)
                
                elif level > 2:
                    new_sub = Section(title=title, level=level, content=[])
                    target = current_section if current_section else current_chapter
                    if target:
                        target.content.append(new_sub)
                        current_section = new_sub 
                else:
                    # Fallback for headings without parent context
                    if not current_chapter:
                        current_chapter = Chapter(title="Untitled Chapter", content=[])
                        chapters.append(current_chapter)
                    fallback_sec = Section(title=title, level=level, content=[])
                    current_chapter.content.append(fallback_sec)
                    current_section = fallback_sec

            # --- HANDLE PARAGRAPHS (Text Content) ---
            elif isinstance(node, Paragraph):
                text_val = self._get_text(node).strip()
                if not text_val:
                    continue
                
                block = ContentBlock(type="text", value=text_val)
                
                if current_section:
                    current_section.content.append(block)
                elif current_chapter:
                    current_chapter.content.append(block)
                else:
                    if not chapters:
                        current_chapter = Chapter(title="Introduction", content=[])
                        chapters.append(current_chapter)
                    current_chapter.content.append(block)

            # --- HANDLE LISTS ---
            elif isinstance(node, List):
                items = []
                for item in node.children:
                    if isinstance(item, ListItem):
                        items.append(self._get_text(item).strip())
                
                if items:
                    block = ContentBlock(type="list_container", value=" | ".join(items))
                    target = current_section if current_section else current_chapter
                    if target:
                        target.content.append(block)

        doc_title = chapters[0].title if chapters else "Extracted Document"
        return DocumentAST(title=doc_title, chapters=chapters)

# --- 3. Execution ---

if __name__ == "__main__":
    sample_input = """
Femicide in Zuid-Amerika door de lens van twee literaire werken: kroniek versus roman

Eva Van Hoey

Universiteit Gent

Volgens de Verenigde Naties Vrouwen is Latijns-Amerika, afgezien van oorlogszones, de meest dodelijke regio voor vrouwen (2017); elke dag worden in Latijns-Amerika minstens negen vrouwen vermoord vanwege hun vrouw-zijn (femicide). De voorbije decennia zijn meer en meer feministische groepen opgestaan die het brutale geweld tegen vrouwen op het continent bestrijden (Souza 2019). Het succes van dergelijke groepen illustreert de publieke afkeer van het schrijnende fenomeen van gendergeweld en meer bepaald femicide, wat ook volop gethematiseerd wordt in de literaire productie van Argentinië en Chili. De Argentijnse Selva Almada en de Chileen Diego Zúñiga publiceerden in 2014 respectievelijk *Chicas muertas*, een kroniek, en *Racimo*, een roman, waarin ze allebei het buitensporige geweld tegen vrouwen centraal stellen.<sup>1</sup>

<sup>1</sup> *Chicas muertas* werd al vertaald in het Nederlands (*Dode meisjes*, Vleugels, vertaald door Marijke Arijs). Andere titels van Selva Almada zijn *Het onweer* en *De steenbakker van El Chaco* (Meulenhoff, vertaald door Adri Boon). *Racimo* werd tot nu toe nog niet

*Chicas muertas* (2014) gaat over drie femicides die in de jaren tachtig in het binnenland van Argentinië plaatsvonden: Andrea Danne, María Luisa Quevedo en Sarita Mundín werden vermoord enkel en alleen vanwege hun vrouw-zijn. De verteller, een alter ego van de auteur zelf, onderzoekt de drie onopgeloste femicides. Ze interviewt niet alleen familieleden maar onderzoekt ook politierapporten in de hoop dichter te komen bij de ontmaskering van de daders van de gruwelijke misdaden. Almada koppelt de waargebeurde verhalen van de drie meisjes aan haar eigen ervaringen als vrouw in Argentinië. Zo slaagt ze erin de harde realiteit voor de vrouw in Argentinië bloot te leggen. *Racimo* (2014) vertelt het fictieve verhaal van de fotograaf Torres Leiva, die geconfronteerd wordt met het tragische en mysterieuze verhaal van de verdwenen jonge meisjes van Alto Hospicio, een dorpje in Chili. Torres Leiva probeert samen met zijn collega-reporter García te achterhalen wat er echt gebeurd is met de jonge meisjes, die van de ene op de andere dag verdwenen. Net zoals in *Chicas muertas*  is de rode draad het brutale geweld tegen vrouwen vanaf jonge leeftijd.

Selva Almada en Diego Zúñiga zijn niet de eerste Latijns-Amerikaanse auteurs die schrijven over het verschrikkelijke fenomeen van de femicide. In de Latijns-

 in het Nederlands vertaald. Er is wel een andere titel voorhanden van Diego Zúñiga namelijk *Camanchaca* (Karaat, vertaald door Merijn Verhulst).

Amerikaanse literatuur geldt *2666*, Roberto Bolaño's *magnum opus* uit 2004, als het meest spraakmakende werk over gendergeweld. Voor deze roman baseerde de Chileense auteur zich op de femicides die in de jaren negentig voor opschudding zorgden in Ciudad Juárez, een Mexicaanse stad op de grens met de Verenigde Staten. Honderden vrouwen zijn toen verdwenen en verkracht, gemarteld en vermoord teruggevonden in de Mexicaanse woestijn (Franco 2013: 219). Naar aanleiding van die gebeurtenissen heeft de term 'femicidio' (Wright 2011: 707) ingang gevonden en is de aandacht nationaal toegenomen voor deze extreme vorm van gendergeweld (Lagarde y De Los Ríos 2010: XI).<sup>2</sup> Het fenomeen van femicide beperkt zich echter niet tot Mexico. De strijd tegen gendergeweld is een van de grootste uitdagingen waar Latijns-Amerika voor staat.

*Chicas muertas* en *Racimo* hebben al aanleiding gegeven tot verschillende artikels*.* De secundaire bibliografie over Almada's kroniek spitst zich toe op twee aspecten: de combinatie van feiten, getuigenissen en fictionele technieken en de aanwezigheid van verschillende discoursen (Tornero 2020; Nanni 2019; Cabral 2018;

<sup>2</sup> Voor relevant onderzoek over femicide in Mexico verwijs ik naar de studies van Rosa-Linda Fregoso en Cynthia Bejarano (2010), Melissa Wright (2011) en Rita Segato (2013).

Moret 2018; Elizondo Oviedo 2015). *Racimo* werd tot nu toe hoofdzakelijk benaderd vanuit de context van globalisering en neoliberalisme (González González & Candia-Cáceres 2017; Rodríguez 2016) en aan de hand van het 'rizoom'- concept van Deleuze en Guattari (Péndola Ramírez & Landeata Mardones 2018).

Hoewel González González en Candia-Cáceres (2017) zowel *Chicas muertas* als *Racimo* bespreken, bestaat er nog geen vergelijkende analyse van beide werken. Een dergelijke comparatieve insteek biedt een dubbel voordeel. Enerzijds leren we eruit hoe femicide wordt voorgesteld in de hedendaagse Argentijnse en Chileense literatuur. Anderzijds brengt een dergelijke vergelijking aan het licht hoe deze voorstelling al dan niet verschilt naar gelang van het gehanteerde literaire genre – Almada en Zúñiga schreven respectievelijk een kroniek en een roman. De focus op de representatie van gendergeweld in deze regio is interessant aangezien het net in Argentinië is dat de succesvolle feministische beweging 'Ni Una Menos' (Niet één [vrouw] minder), ontstond. Intussen heeft dit collectief het voortouw genomen in de strijd tegen gendergeweld, en specifiek femicide, op het hele continent (Santomaso 2017).

De kroniek is een Latijns-Amerikaans genre dat op dit moment aan een steile opmars bezig is. De hedendaagse kroniek wordt gedefinieerd als een hybride genre dat literatuur (fictie) en journalistiek (feiten) combineert om een alternatief discours te bieden over sociale problemen zoals gendergeweld (Castillo 2015). Vooral de

aanwezigheid van de stem van de auteur aan de hand van een eerstepersoonsverteller is een onderscheidend kenmerk van het genre. In het tweede deel van dit artikel ga ik dieper in op de specificiteit van het Latijns-Amerikaanse genre dat minder algemeen bekend is dan het genre van de roman.

Femicide in *Chicas muertas* (2014) en *Racimo* (2014)

In 1992 formuleerden Diana Russell en Jill Radford een van de eerste definities van 'femicide'. Volgens hen is femicide: 'het vermoorden van vrouwen, door mannen, vanwege hun vrouw-zijn' (Organización de los Estados Americanos: Comisión Interamericana de Mujeres 2008: 3). Verder wordt femicide gekarakteriseerd door het feit dat in theorie alle mannen ertoe in staat zijn (Segato 2013). Het gaat om hele brutale moorden: vrouwen worden verkracht, verminkt en gemarteld. Volgens Rita Segato (2013) wijst dit op de vrouwenhaat die aan de basis ligt van het geweld.

In Latijns-Amerika heerst een machocultuur. Mannen uiten hun dominantie en mannelijkheid door middel van de controle op vrouwen en hun lichamen (Yugueros 2014). Het geweld tegen vrouwen moet volgens Ileana Rodríguez (2009) dan ook in de eerste plaats gezien worden als een uiting van macht en controle. Vooral in *Chicas muertas* komt de dominante houding van mannen duidelijk naar voren. De verteller, het alter ego van de auteur, geeft het voorbeeld van een chauffeur die vond dat hij het recht had om haar en haar vriendin ongewenst aan te raken (Almada 2015: 30-33). Vrouwen

die zich verzetten tegen de mannelijke dominantie, worden gestraft. Almada vertelt over een meisje dat verkracht werd met een fles nadat ze een jongen had afgewezen (Almada 2015: 20). Verder benadrukt ze dat femicide niet de enige vorm van gendergeweld is maar dat er ook subtiele varianten voorkomen:

De mama van mijn vriendin die zich niet opmaakte omdat haar papa dat niet toeliet. De collega van mijn mama die elke maand haar volledige loon afstond aan haar echtgenoot opdat hij het kon beheren. […] Zij die geen hakken mocht dragen omdat dat iets voor hoeren is.<sup>3</sup> (Almada 2015: 55-56). Eigen vertaling.

In *Racimo* zijn het voornamelijk de autoriteiten die blijk geven van machogedrag. Volgens de agenten die de verdwijning van de jonge meisjes onderzoeken, kunnen ze niet ontvoerd zijn omdat ze al seksueel actief waren (Zúñiga 2015: 129), wat een duidelijk voorbeeld is van *victim blaming*. Er is echter een opvallend verschil tussen *Chicas muertas* en *Racimo*. In haar kroniek benadrukt Almada heel sterk de link tussen

<sup>3</sup> 'La mamá de mi amiga que no se maquillaba porque su papá no la dejaba. La compañera de trabajo de mi madre que todos los meses le entregaba su sueldo completo al esposo para que se lo administrara. […] La que tenía prohibido usar zapatos de taco porque eso era de puta.' (Almada 2015: 55-56). Originele tekst in het Spaans.

gendergeweld en de misogyne houding van mannen tegenover vrouwen in Latijns-Amerika, iets wat bij Zúñiga veel minder het geval is.

Almada wil nadrukkelijk bestaande overtuigingen over gendergeweld ontkrachten. Femicides worden namelijk in de pers vaak voorgesteld als geïsoleerde incidenten, als uitgevoerd door psychopaten of als passiemoorden (Santomaso 2017). In haar kroniek toont Almada daarentegen de structurele kant van het geweld tegen vrouwen. Ze beklemtoont het hallucinante aantal femicides en het veelvuldig voorkomen van gendergeweld en laat zien dat het niet enkel over psychopaten in donkere steegjes gaat: 'Ik ben nu veertig en, in tegenstelling tot haar en de duizenden vermoorde vrouwen in ons land sinds haar dood, ben ik nog in leven. Alleen maar een kwestie van geluk.'<sup>4</sup> (Almada 2015: 181-182; eigen vertaling). Verder uit ze onomwonden kritiek op de *mainstream* journalistiek in haar land, die op sensationele en morbide wijze bericht over de femicides:

<sup>4</sup> 'Ahora tengo cuarenta años y, a diferencia de ella y de las miles de mujeres asesinadas en nuestro país desde entonces, sigo viva. Sólo una cuestión de suerte.' (Almada 2015: 181-182). Originele tekst in het Spaans.

De zogenoemde 'zaak Quevedo' […] groeide uit tot dé horrorserie van de zomer van 1984 in El Chaco. Een verhaal van intriges, verdenkingen, valse pistes en valse getuigenissen dat door de mensen gevolgd werd via de kranten en de radio alsof het een soap of feuilleton was.<sup>5</sup> (Almada 2015: 151-152). Eigen vertaling.

Zowel Almada als Zúñiga slagen erin om dit discours te vermijden. Almada bespreekt bijvoorbeeld niet de gewelddaad zelf, maar focust veeleer op het resultaat van de misdaad. Op die manier roept ze de wreedheid van het geweld op zonder het letterlijk te tonen. Beide auteurs spreken op een respectvolle manier over de slachtoffers en besteden aandacht aan het leed van de familieleden van de slachtoffers zonder voyeuristisch te zijn.

In *Racimo* legt Zúñiga de focus vooral op de economische kant van het (gender)geweld. Die economische dimensie van het geweld tegen vrouwen heeft te maken met wat Sayak Valencia Triana (2012) *capitalismo gore* (goor kapitalisme)

 5 'El llamado Caso Quevedo […] [se transformó] en la serie de horror y misterio del verano chaqueño de 1984. Un relato de intrigas, sospechas, pistas falsas y falso testimonio que la gente seguía por los diarios y la radio como si fuera un culebrón o un folletín por entregas.' (Almada 2015: 151-152). Originele tekst in het Spaans.

genoemd heeft, meer bepaald de relatie tussen extreem geweld en de neoliberale logica in Mexico. Volgens Valencia worden lichamen niet enkel gebruikt om producten te maken maar zijn ze producten *an sich* geworden, voornamelijk in grensregio's die het economisch slecht doen en waar veel armoede heerst. Mede door de machocultuur die gereproduceerd wordt door de overheid en haar instellingen ontstaat het idee dat het 'legitiem' is om vrouwen en hun lichamen te gebruiken als waren het (economische) wegwerpproducten. Die logica is zichtbaar in het geval van prostitutie, waar vrouwen letterlijk als product worden verkocht, maar ook in de privésfeer worden vrouwen in Latijns-Amerika verkracht, gedood en daarna op een vuilnisbelt gegooid alsof ze wegwerpproducten zijn.

Ondanks het feit dat ze niet in Mexicaans grensgebied liggen, vertonen de gebieden waar *Chicas muertas* en *Racimo* zich afspelen sterke gelijkenissen met de situatie die Valencia beschrijft.<sup>6</sup> Beide verhalen vinden plaats in perifere regio's die het economisch slecht doen en volledig in de steek gelaten zijn door de overheid (Zúñiga 2015: 185). In *Racimo* gaat het over de woestijnstad Iquique in het noorden van Chili en

<sup>6</sup> In hun artikel over het werk van Bolaño, Almada en Zúñiga gebruiken González González & Candia-Cáceres (2017) Valencia's *capitalismo gore*-concept om over perifere en grenszones te spreken in Mexico, Argentinië en Chili.

in *Chicas muertas* over kleine dorpen gelegen in de provincies Chaco, Entre Ríos en Córdoba in Argentinië. De regio's zijn weliswaar geïndustrialiseerd maar de industrialisatie is er mislukt. Arbeiders zijn er ongelukkig (Zúñiga 2015: 106) en dorpen worden beschreven als vuil en lawaaierig (Almada 2015: 63). Door de moeilijke economische situatie beginnen meisjes op jonge leeftijd hun lichaam te verkopen zoals ook het geval is bij Sarita (Almada 2015: 58). In *Racimo* benadrukt Zúñiga vooral het probleem van de georganiseerde prostitutie die over de landsgrenzen heen bestaat (meer bepaald tussen Chili en Peru). Meisjes worden verhandeld en verplaatst over het hele continent. Een van de hypotheses over wat er met de meisjes is gebeurd, is dan ook dat ze ontvoerd en verkocht zijn aan een internationaal prostitutienetwerk (Zúñiga 2015: 221). Verder worden de vermoorde meisjes vaak teruggevonden op vuilnisbelten (Almada 2015: 18; Zúñiga 2015: 201), wat het idee van een vrouwenlichaam als afval verder versterkt. Hoewel Almada ook aandacht heeft voor de moeilijke economische situatie, is het vooral Zúñiga die de problematische economische logica benadrukt aan de hand van zijn focus op prostitutie.

Een derde aspect van femicide dat in beide werken aan bod komt, is de schrijnende straffeloosheid en het hieruit voortvloeiende gebrek aan gerechtigheid voor de slachtoffers en hun familie. Gabriel Giorgi (2014) stelt dat er in de Latijns-Amerikaanse cultuur een onderscheid wordt gemaakt tussen 'levens om te beschermen' en 'levens om in de steek te laten' (Giorgi 2014: 15). Degenen die tot de tweede categorie behoren, mogen uitgebuit en gebruikt worden als waren het producten en hebben geen sociale of politieke rechten (Giorgi 2014: 15-16). Het vermoorden van iemand uit die categorie wordt dan ook niet als misdaad beschouwd (Giorgi 2014: 22). In Latijns-Amerika lijken vrouwen tot die categorie te behoren. Dat zou verklaren waarom de overheid en haar instellingen niets doen om de femicides te voorkomen, laat staan om de daders te vinden en te straffen. In *Chicas muertas* en *Racimo* is dat effectief het geval en doen de autoriteiten geen enkele moeite om de verantwoordelijken van de gruwelijke femicides op te sporen. Ze verzinnen excuses, communiceren niets aan de families en blijken corrupt te zijn, wat ervoor zorgt dat de bevolking geen enkel vertrouwen meer heeft in de politie noch in de werking van justitie.

Beide auteurs tonen hoe de bevolking reageert op de straffeloosheid. De families gaan zelf op onderzoek om de waarheid te vinden (Almada 2015: 153) door te rade te gaan bij waarzeggers of hun toevlucht te nemen tot andere vormen van populaire religie (Almada 2015: 41; Zúñiga 2015: 130). In *Chicas muertas* wordt het belang van solidariteit tussen vrouwen sterk in de verf gezet: vrouwen moeten openlijk spreken over het geweld (Almada 2015: 56) en mogen het niet doodzwijgen. In *Racimo* ligt de nadruk meer op het belang van publieke actie, meer bepaald het organiseren van publieke demonstraties:

De first lady gaat verder spreken maar het geroep gaat het haar niet toestaan. Zes, zeven, acht, tien personen hebben postgevat tegenover de ingang van het Patricio Lynch-college met borden en ze roepen […]. Ieder heeft om zijn hals de gekopieerde foto van zijn dochter, zijn kleindochter, zijn zus, zijn nicht.<sup>7</sup> (Zúñiga 2015: 161). Eigen vertaling.

Deze demonstraties lijken een expliciete verwijzing naar de Dwaze Moeders van de Plaza de Mayo en de Vereniging van familieleden van de Verdwenenen ('Desaparecidos'), twee bewegingen die in de jaren zeventig ontstonden in Argentinië en Chili en die via demonstraties in opstand kwamen tegen de misdaden tegen de menselijkheid die de militaire dictaturen in beide landen begingen (Horton 2015). In Argentinië gaat het over de militaire junta (1976-1983) met als spilfiguur Jorge Rafael Videla; in Chili over de militaire dictatuur van Augusto Pinochet (1973-1990). Telkens 'verdwenen' duizenden (ongeveer 30 000 in Argentinië en 3000 in Chili) tegenstanders

 7 'Va [la primera dama] a continuar pero los gritos no se lo van a permitir. Son seis, siete, ocho, diez personas que se paran frente a la entrada del Patricio Lynch con pancartas y gritan […]. Cada uno lleva colgada en su cuello la imagen fotocopiada de su hija, de su nieta, de su hermana, de su sobrina.' (Zúñiga 2015: 161). Originele tekst in het Spaans.

van de regimes. Aan de hand van foto's van de vermisten slaagden deze burgerbewegingen erin de regimes onder druk te zetten.

Zowel Almada als Zúñiga leggen verbanden tussen de repressie door de staat tijdens de dictaturen en de hachelijke situatie waarin vrouwen zich vandaag de dag bevinden, waardoor de specifieke context en geschiedenis van respectievelijk Argentinië en Chili nauw met de verhalen verweven worden. Zo creëren ze een continuïteit met het verleden. Ze suggereren dat, ondanks de huidige democratie in beide landen, sommige problemen zoals het extreme gendergeweld daarvan een uitvloeisel zijn. Tevens vergemakkelijken ze het identificatieproces bij die lezers die minder vertrouwd zijn met gendergeweld, maar wel opgroeiden tijdens de gruwelijkheden van de militaire dictaturen.

*Chicas muertas* speelt zich bijvoorbeeld af ten tijde van de inauguratie van Raúl Alfonsín, de eerste democratisch verkozen president na de dictatuur van Videla. Almada bespreekt ook verschillende gruweldaden die plaatsvonden tijdens de dictatuur zoals het verminken van dode lichamen en de illegale toe-eigening van baby's (Almada: 151). Het is echter vooral *Racimo* dat bol staat van de verwijzingen naar de dictatuur, in casu naar het regime onder Pinochet. Iquique, de stad in het noorden van Chili waar de meisjes verdwenen, werd bijvoorbeeld de lievelingsstad van Pinochet genoemd. Niet alleen roept het hoofdthema, de verdwijning van jonge meisjes, onmiddellijk associaties op met de verdwijningen die plaatsvonden tijdens de militaire dictatuur, maar ook de sterke focus op het belang van foto's is in *Racimo* frappant. Het hoofdpersonage, Torres Leiva, is fotograaf en reflecteert dikwijls over het belang van herinneren en de ethische dimensie ervan. Zo fotografeert Torres Leiva de demonstraties van de familieleden van de verdwenen meisjes. In *Racimo* wordt echter ook verwezen naar de kracht van beelden in het algemeen. Zo is het eerste deel van de naam van het hoofdpersonage 'Torres', wat 'torens' betekent, een verwijzing naar een van de bekendste voorbeelden van de impact van beelden: de beelden van de aanslagen op 9/11 in de Verenigde Staten, die wereldwijd een enorme invloed hadden:

De man loopt recht naar de camera, zonder zijn tas ook maar een ogenblik los te laten. Torres Leiva kent zijn naam niet, noch zijn leeftijd, noch wat hij daar doet. Hij kan zich echter een verhaal voorstellen, zelfs een naam. […]. Ze lieten hem toe daar te zijn, in dat beeld, met een rookwolk achter zijn rug, verdergaand tussen brokstukken, kijkend naar de grond, zonder ook maar een ogenblik zijn tas los te laten.<sup>8</sup> (Zúñiga 2015: 39). Eigen vertaling.

Ook de werking van het geheugen en de rol die foto's spelen in het herinneringsproces komen uitgebreid aan bod in de roman, net zoals de fragiliteit van het geheugen en het gevaar van vergeten. Het thema 'vergeten' wordt vooral op symbolische wijze opgevoerd wanneer aan het einde van de roman een enorme vloedgolf Iquique overspoelt. Via de problematiek van gendergeweld, die steeds meer aandacht krijgt in Latijns-Amerika, en de verbanden die worden gelegd met de militaire dictatuur van de vorige eeuw slaagt *Racimo* erin de lezer te doen stilstaan bij het belang van het levend houden van herinneringen aan het verleden om het heden beter te begrijpen.

Torres Leiva no sabe su nombre, tampoco su edad ni qué hace ahí. Sin embargo, puede imaginar una historia, un nombre, incluso. […] le permitieron estar ahí, en esa imagen, con una nube de humo a su espalda, avanzando entre escombros, mirando al suelo, sin soltar en ningún momento su maletín.' (Zúñiga 2015: 39). Originele tekst in het Spaans.

<sup>8</sup> 'El hombre camina directo a la cámara, sin soltar en ningún momento el maletín.

## Kroniek versus roman

Uit wat voorafgaat, kunnen we afleiden dat de belangrijkste kenmerken van gendergeweld in het algemeen en femicide in het bijzonder in beide werken op de voorgrond treden. Toch blijkt duidelijk dat de twee auteurs dit geweld op een andere manier benaderen en verschillende aspecten ervan beklemtonen. Hoewel dat verschil in benadering niet te herleiden valt tot een enkele oorzaak wil ik in dit deel van mijn artikel ingaan op een van de factoren die een invloed heeft gehad op de voorstelling van het geweld.<sup>9</sup> Mijn hypothese is dat het verschil in benadering gedeeltelijk samenhangt met het genre dat wordt gebruikt, respectievelijk de kroniek (*Chicas muertas*) en de roman (*Racimo*).

In wat volgt, zal ik me vooral toespitsen op de eigenschappen van de hedendaagse Latijns-Amerikaanse kroniek, een genre waarmee de Nederlandstalige lezer veel minder vertrouwd is. De Latijns-Amerikaanse kroniek in haar huidige vorm groeide uit tot een

 <sup>9</sup> Een andere factor die een invloed kan hebben op de voorstelling van het gendergeweld is het geslacht van de auteurs. Het is bijvoorbeeld mogelijk dat Almada meer aandacht besteedt aan de machocultuur in Latijns-Amerika omdat ze als vrouw al vaker geconfronteerd werd met de nefaste gevolgen ervan.

echt fenomeen sinds de jaren negentig van de vorige eeuw (Darrigrandi 2013) en wordt gekarakteriseerd als een bij uitstek hybride genre (Castillo 2015) dat zich situeert op het raakvlak tussen journalistiek en literatuur, feiten en fictie.<sup>10</sup> Het hybride karakter van het genre wordt benadrukt in de vele 'definities' die erover circuleren. In 2006 definieerde Juan Villoro de hedendaagse Latijns-Amerikaanse kroniek als 'het vogelbekdier van het proza' ('el ornitorrinco de la prosa'). In zijn vaak geciteerde beschrijving stelt de Mexicaanse auteur dat de kroniek elementen uit de roman, de reportage, het interview, het moderne en Griekse theater, het essay en de autobiografie gebruikt zonder samen te vallen met een van die genres. Een opvallende eigenschap van de hedendaagse kroniek is de aanwezigheid van een eerstepersoonsverteller: het objectieve karakter van de feiten en de hoge graad van referentialiteit worden gecombineerd met de subjectiviteit van de eerste persoon en bepaalde fictionaliseringstechnieken.

<sup>10</sup> De kroniek als literaire vorm heeft al een lange traditie. Aan het einde van de negentiende eeuw maakte de modernistische kroniek opgang. Verder was er in de jaren zeventig van de twintigste eeuw in Latijns-Amerika een opvallende opkomst van de stadskroniek. Bovendien publiceerden verschillende auteurs meesterlijke kronieken buiten deze perioden, zoals de Argentijnse auteur Rodolfo Walsh met zijn kroniek *Operación masacre* (1957).

Verder wordt de hedendaagse Latijns-Amerikaanse kroniek beschouwd als een genre met een emancipatorisch potentieel. De kroniek wil een alternatief discours brengen over sociale problemen in de maatschappij en gaat hiermee in tegen hegemonische discoursen in Latijns-Amerika, namelijk die vertogen die gecontroleerd worden door de overheid of door grote mediaconcerns (Castillo 2015). Deze instanties nemen, in Latijns-Amerika nog meer dan elders, vaak een loopje met de objectieve 'waarheid' en hebben de neiging de machocultuur, die ingebakken zit in de maatschappij, te reproduceren en uit te vergroten. Een representatief voorbeeld van de soms problematische berichtgeving over gendergeweld in Latijns-Amerikaanse kranten is het artikel 'Een fanatieke fan van de discotheek en schoolverlater' ('Una fanática de los boliches, que abandonó la secundaria') gepubliceerd op 13 september 2014 in *Clarín*, de Argentijnse krant met de grootste oplage. In dat artikel, dat handelt over de verdwijning van een jong meisje, wordt vooral aandacht gevraagd voor het gedrag van het meisje (ze ging vaak uit en hing rond met oudere jongens), waardoor geïnsinueerd wordt dat ze zelf mede verantwoordelijk is voor wat haar is overkomen.

Aangezien de eigenschappen van de roman afdoende bekend zijn, beperk ik mij hier tot een samenvatting ervan zoals beschreven in het 'Algemeen letterkundig lexicon' (DBNL): het genre van de roman verwijst naar (proza)teksten die overwegend fictioneel zijn en in omvang langer zijn dan novelle teksten en daarom op thematisch vlak vaak

een complexer plot hebben dat bestaat uit hoofd- en nevenintriges. Het is echter uiteraard zo dat deze eigenschappen niet alleen 'voorbehouden zijn aan de roman' (DNBL). Voor het opzet van dit artikel is vooral het overwegend fictionele karakter en de thematische complexiteit (het 'breder van opzet zijn') van de roman interessant aangezien het genre op deze twee niveaus afwijkt van de hedendaagse Latijns-Amerikaanse kroniek, die een sterke non-fictionele inslag heeft en een relatief eenduidige boodschap heeft.

De nadruk die Almada legt op de structurele dimensie van het geweld tegen vrouwen, de heersende machocultuur, die ook op het niveau van de autoriteiten bestaat, en de problematische weergave van het geweld door de mainstream media in Argentinië valt direct in verband te brengen met het genre dat ze gebruikt om het te hebben over gendergeweld in het algemeen en femicide in het bijzonder. Met haar kroniek *Chicas muertas* wil Almada duidelijk een ander discours brengen over geweld tegen vrouwen. Aan de hand van getuigenissen van familieleden en anekdotes uit haar eigen leven, die ze in de eerste persoon vertelt, schetst ze een beeld van de jonge meisjes dat haaks staat op het beeld dat gecreëerd wordt in de mainstream media en in het officiële discours, dat ze weergeeft in de kroniek aan de hand van autopsierapporten en rechtsverslagen. Door het hybride karakter van haar tekst kan ze de slachtoffers van gendergeweld een stem en een gezicht geven en ingaan tegen heersende overtuigingen over het geweld.

Ze toont dat gendergeweld een realiteit is of kan zijn voor alle vrouwen en dat het niet gaat om geïsoleerde incidenten door vreemden: 'Ze hebben ons nooit verteld dat je echtgenoot, je papa, je broer, je neef, je buur, je grootvader, je leerkracht je kon verkrachten. Een man in wie je al je vertrouwen stelde.'<sup>11</sup> (Almada 2015: 55; eigen vertaling). De wens van de auteur om een alternatief discours te brengen, in dit geval over gendergeweld, en om het verhaal op zichzelf te betrekken, is bij uitstek een eigenschap van de kroniek, en daarom veel sterker aanwezig in *Chicas muertas* dan in de roman *Racimo*.

In *Racimo* verbindt Zúñiga de specifieke problematiek van gendergeweld niet alleen met het geweld uit het verleden, meer bepaald dat van de militaire dictatuur, en met de economische achterstelling van bepaalde Chileense regio's maar ook met het 'filosofische' belang van herinneren en het gevaar van vergeten. Anders dan bij de kroniek, hoeft Zúñiga zich in zijn roman niet aan concrete feiten te houden noch een alternatief discours te brengen, wat hem een grotere spanbreedte verleent. Die vrijheid wordt weerspiegeld in de ruime waaier aan thema's die Zúñiga expliciet en impliciet,

<sup>11</sup> 'Nunca nos dijeron que podía violarte tu marido, tu papá, tu hermano, tu primo, tu vecino, tu abuelo, tu maestro. Un varón en el que depositaras toda tu confianza.' (Almada 2015: 55). Originele tekst in het Spaans.

aan de hand van symboliek, de revue laat passeren in zijn roman. Verder is er naast de hoofdintrige, de verdwijning van jonge meisjes, ook sprake van verschillende nevenintriges, zoals het verleden van Torres Leiva en de dood van zijn dochter. Zúñiga baseerde zijn fictionele tekst *Racimo* op het waargebeurde verhaal van Julio Pérez Silva, de seriemoordenaar van Alto Hospicio die actief was tussen 1998 en 2001 (Péndola en Landeata 2018: 36). Door te kiezen voor een roman lijkt het alsof Zúñiga de realiteit gefictionaliseerd heeft om er ruimer over na te kunnen denken.

Vooral het verband tussen fotografie en herinneren is heel belangrijk in deze roman. Ook Zúñiga's focus op de economische kant van het gendergeweld gaat terug op het belang dat hij in zijn roman toekent aan de 'herinnering' van het verleden. De huidige economische situatie van de Chilenen is namelijk mee het gevolg van het Pinochet-regime: de Chilenen in en buiten de steden zijn het slachtoffer van de enorme ongelijkheid tussen arm en rijk en van het doorgeschoten neoliberalisme dat begon onder Pinochet. Dat die problemen leven bij de Chileense bevolking werd duidelijk tijdens de recente opstanden in Chili (2019) tegen de verhoging van de prijs van de metrotickets. Of zoals Zúñiga het zelf verwoordde: 'De overgang [naar de democratie]

leek heel ordelijk en leek een welvarend land op te leveren, maar het ging niet zo goed met ons.'<sup>12</sup> (Querol 2015; eigen vertaling).

Zijn verteller laat in het midden of de meisjes ontvoerd werden door een psychopaat of door een internationaal prostitutienetwerk, wat contrasteert met de boodschap van Almada die net wil benadrukken dat we te maken hebben met een structureel probleem en dat iedereen in staat is tot geweld tegen vrouwen. Zúñiga gebruikt de vrijheid die de roman hem biedt om het probleem van gendergeweld open te trekken naar een filosofische reflectie over de relaties tussen het heden en het verleden terwijl Almada aan de hand van de kroniek de waargebeurde feiten aangrijpt om te reageren tegen het hegemonische discours over het geweld in Latijns-Amerika.

Beide werken vertonen desondanks ook sterke gelijkenissen. Net zoals Zúñiga verwijst Almada naar de verdwijningen tijdens de dictatuur door erop te wijzen dat het lichaam van Sarita nooit gevonden werd. En van zijn kant stelt Zúñiga het schrijnende geweld tegen vrouwen meermaals aan de kaak: 'Ze [de jonge meisjes] leerden sneller dan wie ook om zich wantrouwig op te stellen tegenover hun klasgenoten, hun broers,

 <sup>12</sup> 'La transición pareció muy ordenada y que dejaba un país próspero, pero no estábamos tan bien.' (Ricardo De Querol 2015). Originele tekst in het Spaans.

hun vaders, hun moeders, de buur die hen af en toe uitnodigde om uit te gaan.'<sup>13</sup> (Zúñiga 2015: 106; eigen vertaling). Verder komt ook het belang van foto's en herinnering aan bod in *Chicas muertas*, zij het wel veel minder expliciet. Almada beschrijft bijvoorbeeld telkens de foto's van de dode meisjes die in de huizen van hun families hangen (Almada 2015: 124).

## Conclusie

Een diepgaande literaire analyse toont aan dat het schrijnende fenomeen van femicide een belangrijke rol speelt in zowel *Chicas muertas* van Selva Almada als *Racimo* van Diego Zúñiga. Beide werken hebben met elkaar gemeen dat ze analoge aspecten van femicide in de concrete context van respectievelijk Argentinië en Chili bespreken door bijvoorbeeld naar de gewelddadige geschiedenis, de militaire dictaturen, van de regio te verwijzen. Dat is een groot verschil met de literaire productie die op Mexico focust, zoals de bekende en dikwijls bestudeerde roman *2666* van Roberto Bolaño. Mexico heeft immers nooit een formele dictatuur gekend en de problemen van het land situeren zich op andere vlakken, zoals de migratie door de lange grens met de Verenigde Staten.

<sup>13</sup> 'Aprendieron […] más rápido que nadie a desconfiar: de sus compañeros, de sus hermanos, de sus padres, de sus madres, del vecino que a veces les invitaba a salir.' (Zúñiga 2015: 106). Originele tekst in het Spaans.

Toch zijn er ook duidelijke verschillen vast te stellen. Terwijl Almada een minutieus beeld schetst van de machocultuur in Latijns-Amerika en het problematische discours van de media over gendergeweld benadrukt, legt Zúñiga eerder de nadruk op de problematische logica van het *capitalismo gore* en het belang van 'herinneren'. Beide auteurs delen dan weer een wereldbeeld waarin geld macht geeft om vrouwen en meisjes te gebruiken als wegwerpproducten. Verder formuleren ze alle twee hevige kritiek op het beleid in Argentinië en Chili en op de instellingen die de scheef gegroeide situatie passief en actief in stand houden. Hoewel Almada ook naar de dictatuur van de vorige eeuw verwijst, is het vooral Zúñiga die expliciet verbanden legt tussen de militaire dictatoriale regimes en de huidige overheid en haar instellingen, omdat zowel toen als nu 'levens om te beschermen' onderscheiden worden van 'levens om in de steek te laten'. Die verbanden die de auteur legt, laten ook een uitgebreide reflectie toe over herinneren en vergeten. Almada schuift vrouwelijke solidariteit naar voren als oplossing, terwijl Zúñiga vooral het belang van publieke actie benadrukt.

Deze verschillen kunnen tot op zekere hoogte in verband worden gebracht met de literaire genres die beide auteurs hebben gehanteerd om het over hetzelfde probleem in dezelfde regio te hebben. Almada schreef een kroniek, een genre dat in toenemende mate gebruikt wordt om reële maatschappelijke problemen op alternatieve wijze voor te stellen. Als een echte chroniqueur toont ze een deel van de werkelijkheid en formuleert

ze een vertoog dat tegen het algemene discours over gendergeweld ingaat. Zúñiga gebruikt dan weer de vrijheid van de roman om de problematiek van femicide open te trekken naar andere verhalen en problemen zoals de nasleep van de dictatuur van Pinochet in Chili en het belang van de herinnering en de rol van fotografie in dit proces. Zúñiga toont zich wel ook kritisch voor de manier waarop de autoriteiten omgaan met het geweld tegen vrouwen en Almada verwijst eveneens naar het belang van foto's voor het levend houden van de herinnering aan de doden.

Bibliografie

Primaire bronnen

Almada, S. *Chicas muertas*. Buenos Aires: Literatura Random House, 2015.

Zúñiga, D. *Racimo*. Barcelona: Literatura Random House, 2015.

Secundaire bronnen

Cabral, M. C. '*Chicas muertas* de Selva Almada. Nuevas formas de la memoria sobre el femicidio en la narrativa Argentina', in: *Orbis Tertius*. 23 (28), 2018, 94–114.

Castillo, A. 'The New Latin American Journalistic *Crónica*, Emotions and Hidden Signs of Reality', in: *Global Media Journal.* 13 (1), 2015.

Clarín.com. 'Una fanática de los boliches, que abandonó la secundaria.', *Clarín.* (13

september 2014) [februari 2020]:

[https://www.clarin.com/policiales/fanatica-boliches-abandono-](https://www.clarin.com/policiales/fanatica-boliches-abandono-secundaria_0_S1ek3YcD7g.html)

[secundaria\\_0\\_S1ek3YcD7g.html.](https://www.clarin.com/policiales/fanatica-boliches-abandono-secundaria_0_S1ek3YcD7g.html)

Darrigrandi, C. 'Crónica latinoamericana: algunos apuntes sobre su estudio', in: *Cuadernos de literatura.* 17 (34), 2013, 122-143.

Digitale bibliotheek voor de Nederlandse letteren (DBNL). 'Algemeen letterkundig lexicon: roman'. *DBNL.* [maart 2020]:

[https://www.dbnl.org/tekst/dela012alge01\\_01/dela012alge01\\_01\\_02737.php](https://www.dbnl.org/tekst/dela012alge01_01/dela012alge01_01_02737.php)*.*

Elizondo Oviedo, M. V. 'Femicidio y exhumación del archivo en *Chicas muertas* de Selva Amada', in: *IV Congreso Internacional Cuestiones críticas*, Rosario, 1 October 2015.

Franco, J. *Cruel Modernity*. Durham: Duke University Press, 2013.

Fregoso R. & C. Bejarano. *Terrorizing Women: Feminicide in the Américas.* Durham: Duke University Press, 2010.

Giorgi, G. *Formas comunes: Animalidad, cultura, biopolítica.* Buenos Aires: Eterna Cadencia, 2014.

González González, D. en A. Candia-Cáceres. 'Geografías invisibles de la globalización: Bolaño, Almada y Zúñiga', in: *Anales de literatura chilena.* 28, 2017, 79-94.

Horton, L. 'Women's Movements in Latin America', in: Almeida, P. & Cordero Ulate, A. (red.) *Handbook of Social Movements across Latin America.* Dordrecht: Springer, 2015, 79-87.

Lagarde y De Los Ríos, M. 'Preface: Feminist Keys for Understanding Feminicide: Theoretical, Political, and Legal Construction' in: Fregoso, R. & Bejarano C. (red.) *Terrorizing Women: Feminicide in the Américas*. Durham: Duke University Press, 2010, XI-XXV.

Moret, Z. 'La imposibilidad de la verdad en *Chicas muertas* de Selva Almada', in: *Letras Femeninas.* 43 (2), 2018, 84–94.

Nanni, S. 'Violencia y resistencia en la voces emergentes de América Latina: *Las chicas muertas* de Selva Almada', in: *Altre Modernitā.* 21, 2019*,* 79-91.

Organización de los Estados Americanos: Comisión Interamericana de Mujeres (OEACIM). 'Declaración sobre el Femicidio.' Aprobada en la Cuarta Reunión del Comité de Expertas/os (CEVI), celebrada el 15 de agosto de 2008.

Péndola Ramírez P.A. & Landeata Mardones P.A. '*Racimo*, la novela rizoma de Diego Zúñiga', in: *Literatura y lingüística.* 38, 2018, 35-53.

Querol, R. de. 'Los niños de la represión chilena llenan los silencios', *El País,* (13 juli 2015) [april 2019]:

[https://elpais.com/cultura/2015/06/09/babelia/1433843677\\_532023.html.](https://elpais.com/cultura/2015/06/09/babelia/1433843677_532023.html)

Rodríguez, F.A. 'Cuerpo y capitalismo: el trabajo de la violencia y el miedo', in:

*Estrategias – Psicoanálisis y Salud Mental.* III.4, 2016, 43-46.

Rodríguez, I. *Liberalism at Its Limits: Crime and Terror in the Latin American Cultural Text.* Pittsburgh: University of Pittsburgh Press, 2009.

Russell, D. E. H. & Radford, J. *Femicide: the Politics of Woman Killing*. Buckingham: Open University Press, 1992.

Santomaso, A. 'Argentina's Life-or-Death Women's Movement.' *Jacobin*, (3 juli 2017) [april 2019]:

<https://jacobinmag.com/2017/03/argentina-ni-una-menos-femicides-women-strike/>*.*

Segato, R. L. *La escritura en el cuerpo de las mujeres asesinadas en Ciudad Juárez.* Buenos Aires: Tinta Limón, 2013.

Souza. N. M. F. de. 'When the Body Speaks (to) the Political: Feminist Activism in Latin America and the Quest for Alternative Democratic Futures', in: *Contexto Internacional.* 41 (1), 2019, 89-112.

Tornero, A. 'Feminicidio, literatura testimonial y yo autoral en Chicas muertas', in: *Confabulaciones. Revista de literatura argentina*. 2, 2020, 39–57.

Valencia Triana, S. 'Capitalismo Gore y necropolítica en México contemporáneo', in: *Relaciones Internacionales.* 19, 2012, 83-102.

Verenigde Naties Vrouwen. 'Latinoamérica es la región más peligrosa del mundo para

las mujeres,' *Objetivos de desarrollo sostenible*, (25 november 2017) [maart 2020]: [https://www.un.org/sustainabledevelopment/es/2017/11/latinoamerica-es-la-region](https://www.un.org/sustainabledevelopment/es/2017/11/latinoamerica-es-la-region-mas-peligrosa-del-mundo-para-las-mujeres/)[mas-peligrosa-del-mundo-para-las-mujeres/.](https://www.un.org/sustainabledevelopment/es/2017/11/latinoamerica-es-la-region-mas-peligrosa-del-mundo-para-las-mujeres/)

Villoro, J. 'La crónica, ornitorrinco de la prosa', *La Nación*, (22 januari 2006) [januari 2020]:

[https://www.lanacion.com.ar/cultura/la-cronica-ornitorrinco-de-la-prosa-nid773985.](https://www.lanacion.com.ar/cultura/la-cronica-ornitorrinco-de-la-prosa-nid773985)

Wright, M. W. 'Necropolitics, Narcopolitics, and Femicide: Gendered Violence on the Mexico-U.S. Border', in: *Signs: Journal of Women in Culture and Society.* 36 (3), 2011, 707-731.

Yugueros, A. 'La violencia contra las mujeres: conceptos y causas', in: BARATARIA. *Revista Castellano-Manchega de Ciencias Sociales.* 18, 2014, 147-159.
"""
    converter = MarkerASTConverter()
    try:
        final_ast = converter.convert(sample_input)
        # model_dump_json() is the Pydantic V2 way
        json_output = final_ast.model_dump_json(indent=4)
        
        with open("data/test/output_ast.json", "w") as f:
            f.write(json_output)
            
        print("Success! JSON saved to output_ast.json")
        print(json_output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()